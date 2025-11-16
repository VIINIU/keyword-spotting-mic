import numpy as np
import os

# --- 1. Verilog와 동일한 설정 ---

INPUT_DIR = "./fpga_weights_qat/" 
SPIKE_FILE_PATH = "C:/vini_dir/kws_mic/spike_stimulus_GOOD_2.txt"

# Q-Format
QW_N = 9
WEIGHT_SCALE_FACTOR = 2**QW_N # 512
QT_N = 11
THRESH_SCALE_FACTOR = 2**QT_N # 2048

# 16비트 Signed Integer
INT16_MIN = -32768
INT16_MAX_SIGNED = 32767 # 2^15 - 1

# LIF 파라미터 (Verilog와 일치)
BETA_Q0_16 = 62259 
THRESHOLD_Q5_11 = 0x0400 # 1024 (Decimal)

# THRESHOLD 정렬 (수정된 Verilog 로직 기준)
THRESHOLD_ALIGNED = THRESHOLD_Q5_11 << (QT_N - QW_N) # 1024 << 2 = 4096

# --- 2. Verilog ROM/BRAM 연산 모방 함수 ---

def float_to_fixed_point_int(float_val: float, scale_factor: int) -> int:
    scaled_val = float_val * scale_factor
    rounded_val = np.round(scaled_val)
    clipped_val = np.clip(rounded_val, INT16_MIN, INT16_MAX_SIGNED)
    return int(clipped_val)

def load_weights_as_int(npy_file: str):
    f_path = os.path.join(INPUT_DIR, npy_file)
    data_float = np.load(f_path)
    int_converter = np.vectorize(lambda f: float_to_fixed_point_int(f, WEIGHT_SCALE_FACTOR))
    return int_converter(data_float).astype(object) 

def read_spike_file(file_path, T_max=2):
    spikes = []
    with open(file_path, 'r') as f:
        for t in range(T_max):
            hex_spike_str = f.readline().strip()
            if not hex_spike_str:
                break
            bin_spike_str = bin(int(hex_spike_str, 16))[2:].zfill(20)
            x_spike_vector = [int(bit) for bit in reversed(bin_spike_str)]
            spikes.append(x_spike_vector)
    return spikes

# (이전 스크립트와 동일한 함수들)
def lif_neuron_step(mem_in_64bit, cur_in_32bit):
    mem_in_py = int(mem_in_64bit)
    beta_extended_py = int(BETA_Q0_16)
    mem_decay_intermediate_py = mem_in_py * beta_extended_py
    mem_decay_33bit_py = mem_decay_intermediate_py >> 16
    cur_in_py = int(cur_in_32bit)
    mem_next_33bit_py = mem_decay_33bit_py + cur_in_py
    spk_out = 0
    mem_out_33bit_py = mem_next_33bit_py
    if mem_next_33bit_py > THRESHOLD_ALIGNED:
        spk_out = 1
        mem_out_33bit_py = mem_next_33bit_py - THRESHOLD_ALIGNED
    return np.int64(mem_out_33bit_py), spk_out

def mac_unit_step(spike_vector, weights_j, bias_j):
    acc_int_py = 0
    for i in range(len(spike_vector)):
        if spike_vector[i] == 1:
            acc_int_py += weights_j[i]
    final_cur_in_int_py = acc_int_py + bias_j
    return np.int32(final_cur_in_int_py)

# --- 5. 메인 검증 실행 ---
def verify_l1_j0_steps():
    print("--- 3단계: T=0, T=1 (L1, j=0) 연산 검증 (Python) ---")
    T_MAX_VERIFY = 2 # T=0, T=1
    
    try:
        spikes_T0_T1 = read_spike_file(SPIKE_FILE_PATH, T_max=T_MAX_VERIFY)
        if len(spikes_T0_T1) < T_MAX_VERIFY:
            print(f"🚨 오류: {SPIKE_FILE_PATH}에서 {T_MAX_VERIFY} 타임스텝의 스파이크를 읽을 수 없습니다.")
            return

        W1_int = load_weights_as_int("W1.npy") # (128, 20)
        B1_int = load_weights_as_int("B1.npy") # (128,)
        
        W1_int_j0 = W1_int[0]
        B1_int_j0 = B1_int[0]
        
        L1_j0_MEM_BRAM = np.int64(0) 
        
        print("\nVerilog 로그와 아래 값을 비교하세요:\n")
        
        for t in range(T_MAX_VERIFY):
            spike_vector_in = spikes_T0_T1[t] # 현재 타임스텝 스파이크
            
            mem_in_64bit = L1_j0_MEM_BRAM
            cur_in_32bit = mac_unit_step(spike_vector_in, W1_int_j0, B1_int_j0)
            mem_out_64bit, spk_out = lif_neuron_step(mem_in_64bit, cur_in_32bit)
            L1_j0_MEM_BRAM = mem_out_64bit
            
            # --- D. 결과 출력 (🚨 [수정] C long 오류를 피하기 위해 int() 사용) ---
            print(f"--- [Python T={t}] ---")
            # 33비트 16진수 (Verilog: %h)
            print(f"  > BRAM Read (mem_in): {int(mem_in_64bit) & 0x1FFFFFFFF:09X}")
            # 32비트 16진수 (Verilog: %h)
            print(f"  > MAC Out (cur_in)  : {int(cur_in_32bit) & 0xFFFFFFFF:08X}")
            # 33비트 16진수 (Verilog: %h)
            print(f"  > LIF Out (mem_out) : {int(mem_out_64bit) & 0x1FFFFFFFF:09X}")
            print(f"  > LIF Spike (spk_out): {spk_out}\n")

    except FileNotFoundError as e:
        print(f"🚨 파일 오류: {e}")
    except Exception as e:
        print(f"🚨 알 수 없는 오류: {e}")

if __name__ == "__main__":
    verify_l1_j0_steps()