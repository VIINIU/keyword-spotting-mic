import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import multiprocessing
import torch.nn.functional as F
from typing import Tuple

# ----------------------------------------------------
# A. Fixed Point 양자화 함수 정의
# ----------------------------------------------------
def quantize_qmn(tensor: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    Floating Point 텐서를 Qm.n Fixed Point 포맷으로 양자화합니다.
    1. Scale (2^n 곱하기) -> 2. Rounding -> 3. Clip (범위 제한) -> 4. Unscale
    """
    
    scale_factor = 2**n
    
    # Qm.n 포맷의 최대/최소 값 (m은 Sign bit 포함 총 비트 수, 즉 m-1은 정수 데이터 비트 수)
    # 16bit Signed Fixed Point의 실제 표현 범위: [-2^(m-1), 2^(m-1) - 2^(-n)]
    max_representable_val = 2**(m - 1) - 2**(-n)
    min_representable_val = -2**(m - 1)
    
    # 2. 스케일 및 라운딩
    quantized_tensor = torch.round(tensor * scale_factor)
    
    # 3. 클리핑 (Overflow/Underflow 방지)
    # 정수부 최댓값/최솟값에 맞춰 클리핑
    quantized_tensor = torch.clamp(
        quantized_tensor, 
        min=min_representable_val * scale_factor, 
        max=max_representable_val * scale_factor
    )
    
    # 4. 언스케일링
    return quantized_tensor / scale_factor

# ====================================================
# 1. 하이퍼파라미터 및 경로 설정 (최적화)
# ====================================================
# 🚨🚨🚨 8bit 양자화된 NPY 파일 경로로 수정 (FPGA와 데이터 일치) 🚨🚨🚨
clear_command_npy_folder_path = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed/spike_16bit_regenerated"
neg_command_npy_folder_path = "C:/Users/11e26/Desktop/internship/source/clear_negative_command/spike_16bit_regenerated" 
FPGA_WEIGHTS_DIR = "./fpga_weights/" 

N_MELS = 20 
NUM_HIDDENS_1 = 128  # 🚨 첫 번째 은닉층
NUM_HIDDENS_2 = 128  # 🚨 두 번째 은닉층 추가
NUM_OUTPUTS = 2
BETA = 0.95
THRESHOLD = 0.5      
spike_grad = surrogate.atan()

T_MAX = 3000         
BATCH_SIZE = 64      
NUM_EPOCHS = 50      
LEARNING_RATE = 5e-4
# 🚨 QAT Finetuning을 위한 설정
FINETUNE_EPOCHS = 30       # QAT로 추가 훈련할 에포크 (50보다 짧게)
FINETUNE_LR = LEARNING_RATE / 10.0  # 🚨 더 작은 LR 사용 (예: 5e-5)
PTQ_MODEL_PATH = "./wws_snn_final_weights.pth"   # 🚨 방금 훈련한 PTQ 모델 경로
QAT_MODEL_SAVE_PATH = "./wws_snn_qat_final_weights.pth" # 🚨 QAT 최종 모델 저장 경로
FPGA_QAT_WEIGHTS_DIR = "./fpga_weights_qat/"          # 🚨 QAT FPGA 가중치 저장 경로

# DataLoader 병렬 처리 설정
NUM_WORKERS = multiprocessing.cpu_count() - 1
if NUM_WORKERS < 1: NUM_WORKERS = 1

# beta = 0.95
class SpikeDataset(Dataset):
    def __init__(self, file_paths, labels, T_max=T_MAX, n_mels=N_MELS):
        self.file_paths = file_paths
        self.labels = labels
        self.T_max = T_max
        self.n_mels = n_mels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            spike_data_np = np.load(file_path) 
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            spike_data_np = np.zeros((self.T_max, self.n_mels), dtype=np.float32)

        if spike_data_np.shape[0] > self.T_max:
             spike_data_np = spike_data_np[:self.T_max, :]
        elif spike_data_np.shape[0] < self.T_max:
             padding = np.zeros((self.T_max - spike_data_np.shape[0], self.n_mels), dtype=np.float32)
             spike_data_np = np.vstack([spike_data_np, padding])
        
        data_tensor = torch.as_tensor(spike_data_np, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return data_tensor, label_tensor

# ----------------------------------------------------
# 3. SNN 모델 클래스 (Fixed Point QAT 로직 통합)
# ----------------------------------------------------
class WWS_SNN(nn.Module):
    # 🚨 Fixed Point 포맷 정의 (동일)
    QW_M, QW_N = 7, 9 
    QT_M, QT_N = 5, 11 
    
    # 🚨 is_qat 플래그 추가
    def __init__(self, num_inputs, num_hiddens_1, num_hiddens_2, num_outputs, beta, threshold, spike_grad, is_qat=False):
        super().__init__()
        
        self.is_qat = is_qat
        
        # 🚨 [중요] PTQ 훈련 때처럼 threshold는 'float'로 초기화
        self.float_threshold = threshold 
        quantized_threshold = threshold 
        
        if self.is_qat:
            # QAT 모드일 경우에만 임계값도 양자화
            quantized_threshold = quantize_qmn(torch.tensor(threshold), self.QT_M, self.QT_N).item()
        # 1. 입력층 -> 은닉층 1
        self.fc1 = nn.Linear(num_inputs, num_hiddens_1)
        self.lif1 = snn.Leaky(beta=beta, threshold=quantized_threshold, spike_grad=spike_grad, reset_mechanism="subtract")
        
        # 2. 🚨 은닉층 1 -> 은닉층 2 (새로 추가)
        self.fc2 = nn.Linear(num_hiddens_1, num_hiddens_2)
        self.lif2 = snn.Leaky(beta=beta, threshold=quantized_threshold, spike_grad=spike_grad, reset_mechanism="subtract")
        
        # 3. 🚨 은닉층 2 -> 출력층 (이름 변경 fc2->fc3, lif2->lif3)
        self.fc3 = nn.Linear(num_hiddens_2, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=quantized_threshold, spike_grad=spike_grad, reset_mechanism="subtract")
        
        self.init_state()
        
    def enable_qat(self):
        print("✅ QAT (Quantization-Aware Training) Finetuning 모드를 활성화합니다.")
        self.is_qat = True
        
        # 1. 🚨 [수정] .item()을 제거하여 PyTorch 텐서 자체를 생성
        q_thresh_tensor = quantize_qmn(torch.tensor(self.float_threshold), self.QT_M, self.QT_N)

        # 2. 🚨 [수정] 모델의 파라미터가 있는 device (cpu or cuda)로 텐서를 이동
        #    (self.fc1.weight.device가 현재 모델이 있는 장치를 알려줌)
        device = self.fc1.weight.device 
        q_thresh_tensor = q_thresh_tensor.to(device)
        
        # 3. 텐서를 lif 뉴런의 threshold에 할당 (이제 Type이 맞음)
        self.lif1.threshold = q_thresh_tensor
        self.lif2.threshold = q_thresh_tensor
        self.lif3.threshold = q_thresh_tensor

    def init_state(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky() # 🚨 mem2 추가
        self.mem3 = self.lif3.init_leaky() # 🚨 mem3 (기존 mem2)

    def quantize_parameters(self):
        # FC1 (Q7.9)
        self.fc1.weight.data = quantize_qmn(self.fc1.weight.data, self.QW_M, self.QW_N)
        self.fc1.bias.data = quantize_qmn(self.fc1.bias.data, self.QW_M, self.QW_N)
        
        # 🚨 FC2 (새로 추가) (Q7.9)
        self.fc2.weight.data = quantize_qmn(self.fc2.weight.data, self.QW_M, self.QW_N)
        self.fc2.bias.data = quantize_qmn(self.fc2.bias.data, self.QW_M, self.QW_N)
        
        # 🚨 FC3 (이름 변경) (Q7.9)
        self.fc3.weight.data = quantize_qmn(self.fc3.weight.data, self.QW_M, self.QW_N)
        self.fc3.bias.data = quantize_qmn(self.fc3.bias.data, self.QW_M, self.QW_N)

    # 🚨 forward 함수 인자 및 내부 로직 수정
    def forward(self, x, mem1, mem2, mem3):
        
        if self.is_qat:
            # === QAT (Fake Quantization) 모드 ===
            
            w1 = self.fc1.weight + (quantize_qmn(self.fc1.weight, self.QW_M, self.QW_N) - self.fc1.weight).detach()
            b1 = self.fc1.bias + (quantize_qmn(self.fc1.bias, self.QW_M, self.QW_N) - self.fc1.bias).detach()
            w2 = self.fc2.weight + (quantize_qmn(self.fc2.weight, self.QW_M, self.QW_N) - self.fc2.weight).detach()
            b2 = self.fc2.bias + (quantize_qmn(self.fc2.bias, self.QW_M, self.QW_N) - self.fc2.bias).detach()
            w3 = self.fc3.weight + (quantize_qmn(self.fc3.weight, self.QW_M, self.QW_N) - self.fc3.weight).detach()
            b3 = self.fc3.bias + (quantize_qmn(self.fc3.bias, self.QW_M, self.QW_N) - self.fc3.bias).detach()
            
            # 2. 양자화된 가중치로 직접 연산
            cur1 = F.linear(x, w1, b1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.linear(spk1, w2, b2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = F.linear(spk2, w3, b3)
            spk3, mem3 = self.lif3(cur3, mem3)
            
        else:
            # === 표준 (Float) PTQ 훈련 모드 ===
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
        return spk3, mem1, mem2, mem3

# ----------------------------------------------------
# 4. Main 실행 블록
# ----------------------------------------------------
if __name__=="__main__":
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # === A. 데이터 로드 및 통합 ===
    
    file_paths = []
    labels = [] 

    # 1. Positive (Alexa, Label = 1) 데이터 로드 
    pos_files = glob.glob(os.path.join(clear_command_npy_folder_path, "*.npy"))
    file_paths.extend(pos_files)
    labels.extend([1] * len(pos_files))
    print(f"Positive 샘플 로드 완료: {len(pos_files)}개")

    # 2. Negative (Non-Alexa, Label = 0) 데이터 로드 
    neg_files = glob.glob(os.path.join(neg_command_npy_folder_path, "*.npy"))
    file_paths.extend(neg_files)
    labels.extend([0] * len(neg_files))
    print(f"Negative 샘플 로드 완료: {len(neg_files)}개") 
    
    if not file_paths:
        print("오류: 학습에 사용할 NPY 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        exit()


    # 3. DataLoader 생성
    spike_dataset = SpikeDataset(file_paths, labels, T_max=T_MAX, n_mels=N_MELS)
    train_loader = DataLoader(spike_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              drop_last=True,
                              num_workers=NUM_WORKERS) 
    print(f"총 학습 데이터셋 크기: {len(file_paths)}개 샘플")
    
    # Loss Weight 계산
    pos_count = len(pos_files)
    neg_count = len(neg_files)
    weight_for_neg = pos_count / neg_count

    # === B. 모델 및 PTQ 가중치 로드 ===
    
    class_weights = torch.tensor([weight_for_neg, 1.0], dtype=torch.float32).to(device) 

    # 🚨 is_qat=False (기본값)로 먼저 모델 객체 생성
    net = WWS_SNN(N_MELS, NUM_HIDDENS_1, NUM_HIDDENS_2, NUM_OUTPUTS, BETA, THRESHOLD, spike_grad, is_qat=False).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device) 
    
    # 🚨 1. 이전 PTQ 훈련 가중치 로드
    try:
        net.load_state_dict(torch.load(PTQ_MODEL_PATH))
        print(f"✅ 성공: {PTQ_MODEL_PATH}에서 훈련된 (PTQ) 가중치를 로드했습니다.")
    except Exception as e:
        print(f"🚨 경고: {PTQ_MODEL_PATH} 로드 실패. ({e})")
        print("이 스크립트는 PTQ 훈련이 완료된 후에 실행해야 합니다.")
        exit()
        
    # 🚨 2. QAT Finetuning을 위한 새 옵티마이저 (더 낮은 LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=FINETUNE_LR, betas=(0.9, 0.999))
    
    # 🚨 3. QAT 모드 활성화!
    # 이 함수가 self.is_qat = True로 바꾸고, threshold도 양자화함
    net.enable_qat() 
    
    print(f"QAT Finetuning 준비 완료. (Epochs: {FINETUNE_EPOCHS}, LR: {FINETUNE_LR})")
    
    # === C. SNN 훈련 루프 (QAT 적용) ===

    for epoch in range(FINETUNE_EPOCHS): # 🚨 FINETUNE_EPOCHS 사용
        net.train()
        total_loss = 0
        total_correct = 0
        
        for inputs, targets in train_loader:
            # ... (inputs, targets .to(device)) ...
            net.init_state() 
            total_output_spikes = torch.zeros(inputs.size(0), NUM_OUTPUTS).to(device) 
            optimizer.zero_grad()
            T_max_current = inputs.size(1) 
            
            # 🚨 [중요] net.quantize_parameters()는 절대 호출 금지!
            # `forward` 함수가 내부적으로 STE QAT를 수행함
            
            # 2. 시간 축 (T) 시뮬레이션
            for step in range(T_max_current):
                spk_out, net.mem1, net.mem2, net.mem3 = net(inputs[:, step, :], net.mem1, net.mem2, net.mem3)
                total_output_spikes += spk_out
                
            # 3. 손실 계산 및 역전파
            loss = loss_fn(total_output_spikes, targets)
            loss.backward()
            optimizer.step() # 👈 이 옵티마이저는 32bit 원본 가중치를 업데이트함
            
            total_loss += loss.item()
            
            # 4. 정확도 계산
            _, predicted = torch.max(total_output_spikes, 1)
            total_correct += (predicted == targets).sum().item()

        # 에포크 결과 출력
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / (len(train_loader) * BATCH_SIZE) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Training Accuracy: {avg_acc:.2f}%")

# === D. QAT 학습 결과 저장 및 FPGA 변환용 추출 ===
    
    print(f"\n✅ QAT Finetuning 완료.")
    
    # 🚨 최종 저장 전, 가중치를 다시 한번 양자화하여 FPGA용으로 저장
    # (QAT 훈련으로 미세하게 바뀐 32bit 가중치를 최종 Q7.9로 변환)
    net.quantize_parameters()
    
    torch.save(net.state_dict(), QAT_MODEL_SAVE_PATH) # 🚨 새 경로에 저장
    print(f"\n✅ QAT Finetuned 모델 가중치가 {QAT_MODEL_SAVE_PATH}에 저장되었습니다.")
    
    os.makedirs(FPGA_QAT_WEIGHTS_DIR, exist_ok=True) # 🚨 새 경로에 저장
    
    # 🚨 저장 로직은 동일 (경로만 변경)
    W1 = net.fc1.weight.data.numpy()
    B1 = net.fc1.bias.data.numpy()
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "W1.npy"), W1)
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "B1.npy"), B1)
    
    W2 = net.fc2.weight.data.numpy()
    B2 = net.fc2.bias.data.numpy()
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "W2.npy"), W2)
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "B2.npy"), B2)
    
    W3 = net.fc3.weight.data.numpy()
    B3 = net.fc3.bias.data.numpy()
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "W3.npy"), W3)
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "B3.npy"), B3)
    
    # LIF 파라미터 저장 (QAT이므로 양자화된 threshold 사용)
    lif_params = {
        'BETA_VAL': BETA, 
        'THRESHOLD_VAL': net.lif1.threshold, # 🚨 enable_qat()에서 이미 양자화됨
        'QW_M': WWS_SNN.QW_M,
        'QW_N': WWS_SNN.QW_N,
        'QT_M': WWS_SNN.QT_M,
        'QT_N': WWS_SNN.QT_N,
    }
    np.save(os.path.join(FPGA_QAT_WEIGHTS_DIR, "LIF_params.npy"), lif_params)
    print(f"✅ QAT Finetuned FPGA 가중치가 {FPGA_QAT_WEIGHTS_DIR}에 저장되었습니다.")
    print("작동 완료")