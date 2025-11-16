import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import os 

# ====================================================
# 1. [신규] Verilog의 정수 연산을 위한 헬퍼 함수
# ====================================================

def quantize_to_int(tensor: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    Float 텐서를 Qm.n '정수' 텐서로 변환합니다.
    Scale(2^n)만 하고 Unscale(나누기)을 하지 않습니다.
    """
    scale_factor = 2**n
    INT16_MIN = -32768
    INT16_MAX = 32767
    quantized_tensor = torch.round(tensor * scale_factor)
    quantized_tensor = torch.clamp(
        quantized_tensor, 
        min=INT16_MIN, 
        max=INT16_MAX
    )
    return quantized_tensor.long() 

# 🚨 [신규] tb_snn_core.v의 $readmemh를 모방하는 .txt 로더
def read_spike_stimulus_txt(txt_file_path: str, n_mels: int) -> np.ndarray:
    """
    Verilog의 16진수 스파이크 .txt 파일을 읽어
    (T, N_MELS) 형태의 float32 numpy 배열로 반환합니다.
    """
    spike_vectors = []
    
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        hex_str = line.strip()
        if not hex_str:
            continue
            
        # 1. 16진수 -> 20비트 2진수 문자열
        bin_str = f'{int(hex_str, 16):0{n_mels}b}'
        
        # 2. 2진수 문자열 -> float 리스트
        # (Verilog [19:0] 순서에 맞게 reversed)
        # (모델 입력은 0.0 또는 1.0 float여야 함)
        spike_vector = [float(bit) for bit in reversed(bin_str)]
        spike_vectors.append(spike_vector)
        
    # (T, N_MELS) 형태의 numpy 배열로 반환
    return np.array(spike_vectors, dtype=np.float32)

# ====================================================
# 2. SNN 모델 (Verilog 하드웨어 모방)
# ====================================================
class WWS_SNN_Hardware_Sim(nn.Module):
    # --- Verilog와 100% 동일한 파라미터 ---
    QW_M, QW_N = 7, 9  # 가중치/편향 Q-Format (Q7.9)
    QT_M, QT_N = 5, 11 # 임계값 Q-Format (Q5.11)
    
    VERILOG_BETA_INT = 62259
    VERILOG_BETA_FLOAT = 62259 / 65536.0 # 0.950002...
    VERILOG_THRESH_INT_Q5_11 = 1024       # 16'h0400
    
    # Verilog의 Q-Format에 맞춘 실수 임계값 (버그 수정된 >> 로직)
    VERILOG_THRESH_FLOAT_ALIGNED = (VERILOG_THRESH_INT_Q5_11 >> (QT_N - QW_N)) / (2**QW_N) # 0.5

    def __init__(self, num_inputs, num_hiddens_1, num_hiddens_2, num_outputs, spike_grad):
        super().__init__()
        
        # 1. [수정] Verilog의 수학을 사용하도록 BETA, THRESHOLD 고정
        self.lif1 = snn.Leaky(beta=self.VERILOG_BETA_FLOAT, 
                              threshold=self.VERILOG_THRESH_FLOAT_ALIGNED, 
                              spike_grad=spike_grad, reset_mechanism="subtract")
        
        self.lif2 = snn.Leaky(beta=self.VERILOG_BETA_FLOAT, 
                              threshold=self.VERILOG_THRESH_FLOAT_ALIGNED, 
                              spike_grad=spike_grad, reset_mechanism="subtract")
        
        self.lif3 = snn.Leaky(beta=self.VERILOG_BETA_FLOAT, 
                              threshold=self.VERILOG_THRESH_FLOAT_ALIGNED, 
                              spike_grad=spike_grad, reset_mechanism="subtract")
        
        self.W1_int = None
        self.B1_int = None
        self.W2_int = None
        self.B2_int = None
        self.W3_int = None
        self.B3_int = None

        self.init_state()

    def init_state(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()

    def load_weights_from_pth(self, state_dict, device):
        """
        .pth 파일(float)을 로드한 뒤, Verilog가 사용할 '정수' 가중치로 변환
        """
        self.W1_int = quantize_to_int(state_dict['fc1.weight'], self.QW_M, self.QW_N).to(device)
        self.B1_int = quantize_to_int(state_dict['fc1.bias'], self.QW_M, self.QW_N).to(device)
        self.W2_int = quantize_to_int(state_dict['fc2.weight'], self.QW_M, self.QW_N).to(device)
        self.B2_int = quantize_to_int(state_dict['fc2.bias'], self.QW_M, self.QW_N).to(device)
        self.W3_int = quantize_to_int(state_dict['fc3.weight'], self.QW_M, self.QW_N).to(device)
        self.B3_int = quantize_to_int(state_dict['fc3.bias'], self.QW_M, self.QW_N).to(device)
        print("✅ Python 모델 가중치를 Verilog의 '정수' 포맷으로 변환 완료.")

    # 🚨 [핵심 수정] F.linear()를 Verilog의 '정수 덧셈'으로 대체
    def hardware_true_forward(self, x, mem1, mem2, mem3):
        
        x_int = x.long().unsqueeze(-1) 
        
        # --- 1. L1 (MAC + LIF) ---
        cur1_int = torch.bmm(self.W1_int.unsqueeze(0).repeat(x.size(0), 1, 1), x_int)
        cur1_int = cur1_int.squeeze(-1) + self.B1_int 
        
        # [디버깅] T=0일 때 j=0 값 확인
        if T_STEP_COUNTER == 0:
             print("\n--- 🐍 [PYTHON T=0] Layer 1 (하드웨어 모방) ---")
             print(f"  > Python L1, j=0 cur_in (int): {cur1_int[0, 0].item() & 0xFFFFFFFF :08x}")
             print("--------------------------------------------------\n")

        cur1_float = cur1_int.float() / (2**self.QW_N)
        spk1, mem1 = self.lif1(cur1_float, mem1)
        
        # --- 2. L2 (MAC + LIF) ---
        spk1_int = spk1.long().unsqueeze(-1) # [B, 128, 1]
        cur2_int = torch.bmm(self.W2_int.unsqueeze(0).repeat(x.size(0), 1, 1), spk1_int)
        cur2_int = cur2_int.squeeze(-1) + self.B2_int
        cur2_float = cur2_int.float() / (2**self.QW_N)
        spk2, mem2 = self.lif2(cur2_float, mem2)
        
        # --- 3. L3 (MAC + LIF) ---
        spk2_int = spk2.long().unsqueeze(-1) # [B, 128, 1]
        cur3_int = torch.bmm(self.W3_int.unsqueeze(0).repeat(x.size(0), 1, 1), spk2_int)
        cur3_int = cur3_int.squeeze(-1) + self.B3_int
        cur3_float = cur3_int.float() / (2**self.QW_N)
        spk3, mem3 = self.lif3(cur3_float, mem3)
        
        return spk3, mem1, mem2, mem3

# ====================================================
# 3. 검증 설정 (🚨 .txt 파일로 변경)
# ====================================================
# VERIFY_NPY_FILE = "C:/.../spike_input_16bit_1.npy"
VERIFY_TXT_FILE = "C:/vini_dir/kws_mic/spike_stimulus_GOOD_3_neg.txt" # 🚨 tb_snn_core.v 경로와 일치
QAT_MODEL_PATH = "./wws_snn_qat_final_weights.pth" 

N_MELS = 20 
NUM_HIDDENS_1 = 128
NUM_HIDDENS_2 = 128
NUM_OUTPUTS = 2
spike_grad = surrogate.atan()
T_MAX = 3000

T_STEP_COUNTER = 0 # 🚨 디버깅용 글로벌 카운터

# ====================================================
# 4. 메인 검증 로직 (🚨 .txt 파일 로더로 변경)
# ====================================================
if __name__ == "__main__":
    
    device = torch.device("cpu")
    print(f"검증 시작: {VERIFY_TXT_FILE} (모델: {QAT_MODEL_PATH})")

    # 1. 모델 생성
    net = WWS_SNN_Hardware_Sim(N_MELS, NUM_HIDDENS_1, NUM_HIDDENS_2, NUM_OUTPUTS, spike_grad).to(device)
    
    try:
        # 2. .pth (float) 로드 -> Verilog '정수' 가중치로 변환
        state_dict = torch.load(QAT_MODEL_PATH, map_location=device)
        net.load_weights_from_pth(state_dict, device)
        
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
        exit()
    
    net.eval() # (추론 모드)

    # 3. 🚨 [수정] NPY 로더 대신 TXT 로더 사용
    try:
        spike_data_np = read_spike_stimulus_txt(VERIFY_TXT_FILE, N_MELS)
        print(f"✅ TXT 파일 로드 성공. 총 {spike_data_np.shape[0]} 타임스텝.")
    except Exception as e:
        print(f"🚨 TXT 파일 로드 실패: {e}")
        exit()

    # (패딩/절삭 로직 - 원본 스크립트와 동일)
    if spike_data_np.shape[0] > T_MAX:
         spike_data_np = spike_data_np[:T_MAX, :]
    elif spike_data_np.shape[0] < T_MAX:
         padding = np.zeros((T_MAX - spike_data_np.shape[0], N_MELS), dtype=np.float32)
         spike_data_np = np.vstack([spike_data_np, padding])
         
    data_tensor = torch.as_tensor(spike_data_np, dtype=torch.float32).to(device)
    data_tensor = data_tensor.unsqueeze(0) # (T, N_MELS) -> (Batch=1, T, N_MELS)
    
    # [입력 검증 플래그]
    first_spike_vector_np = spike_data_np[0, :]
    # (float 1.0/0.0 -> int 1/0 변환)
    bin_str = "".join(['1' if x > 0 else '0' for x in first_spike_vector_np])
    hex_str = f'{int(bin_str, 2):05X}'
    print(f"--- 🐍 [PYTHON T=0] Input Spike Vector (from TXT): {hex_str} ---")

    # 4. SNN 추론 (T_MAX 스텝)
    net.init_state()
    total_output_spikes = torch.zeros(1, NUM_OUTPUTS).to(device)
    
    with torch.no_grad():
        for step in range(T_MAX):
            T_STEP_COUNTER = step # 🚨 디버깅 카운터 업데이트
            
            spk_out, net.mem1, net.mem2, net.mem3 = net.hardware_true_forward(
                data_tensor[:, step, :], net.mem1, net.mem2, net.mem3
            )
            total_output_spikes += spk_out

    # 5. 최종 결과 출력
    print("\n================= 🐍 '진짜' 파이썬 정답지 (하드웨어 모방) ==================")
    print(f"  {VERIFY_TXT_FILE} 파일을 사용한 결과:")
    print(f"  최종 누적 스파이크 (0: neg, 1: alexa):")
    print(f"   >> {total_output_spikes[0].cpu().numpy()}")
    print("=======================================================================")