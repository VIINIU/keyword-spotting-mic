import numpy as np
import os

# ----------------------------------------------------
# 1. 설정 (QAT 코드와 일치해야 함)
# ----------------------------------------------------

# QAT 학습이 완료된 가중치(.npy)가 저장된 폴더
INPUT_DIR = "./fpga_weights_qat/"

# Verilog ROM/BRAM이 읽을 .mem 파일들을 저장할 폴더
OUTPUT_DIR = "./verilog_mem_files/"

# 가중치/편향 포맷: Q7.9 (M=7, N=9)
QW_M = 7
QW_N = 9
WEIGHT_SCALE_FACTOR = 2**QW_N # 512

# LIF 임계값 포맷: Q5.11 (M=5, N=11)
QT_M = 5
QT_N = 11
THRESH_SCALE_FACTOR = 2**QT_N # 2048

# 16비트 Signed Integer의 최대/최소값 (np.int16)
INT16_MIN = -32768 # -2^15
INT16_MAX = 32767  # 2^15 - 1

# ----------------------------------------------------
# 2. 핵심 변환 함수
# ----------------------------------------------------

def float_to_fixed_point_int(float_val: float, scale_factor: int) -> int:
    """
    하나의 float 값을 고정 소수점 '정수'로 변환합니다.
    (Python의 np.round 기준)
    """
    
    # 1. 스케일링 (float * 2^N)
    scaled_val = float_val * scale_factor
    
    # 2. 반올림 (np.round: 0.5는 짝수 정수로 반올림. 예: 2.5->2, 3.5->4)
    # 16bit_quant_finetune.py의 quantize_qmn 함수와 동일한 로직
    rounded_val = np.round(scaled_val)
    
    # 3. 16비트 정수 범위로 클리핑 (Saturation)
    clipped_val = np.clip(rounded_val, INT16_MIN, INT16_MAX)
    
    return int(clipped_val)

def convert_npy_to_mem(
    npy_filename: str, 
    mem_filename: str, 
    scale_factor: int,
    output_format: str = 'hex' # $readmemh용
):
    """
    .npy 파일을 로드하여 .mem 파일로 변환합니다.
    """
    input_path = os.path.join(INPUT_DIR, npy_filename)
    output_path = os.path.join(OUTPUT_DIR, mem_filename)
    
    try:
        data_float = np.load(input_path)
    except FileNotFoundError:
        print(f"🚨 경고: '{input_path}' 파일을 찾을 수 없습니다. 건너뜁니다.")
        return None # 🚨 None 반환

    print(f"변환 중: {input_path} -> {output_path}")

    # .mem 파일 쓰기
    with open(output_path, 'w') as f:
        
        # np.nditer를 사용해 다차원 배열도 C-style (row-major) 순서로 순회
        # W3[0,0]...W3[0,127], W3[1,0]...W3[1,127] 순서
        for float_val in np.nditer(data_float):
            
            # Float -> Fixed-Point Int 변환
            int_val = float_to_fixed_point_int(float(float_val), scale_factor)
            
            # 16비트 2의 보수 16진수 (예: -1 -> FFFF, -127 -> FF81)
            # (int_val & 0xFFFF)는 음수를 2의 보수로 자동 변환
            hex_str = f'{(int_val & 0xFFFF):04X}'
            f.write(f"{hex_str}\n")
            
    return data_float # 🚨 검증을 위해 로드된 float 배열 반환

# ----------------------------------------------------
# 3. 메인 실행
# ----------------------------------------------------
if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    weight_files = {
        "W1.npy": "W1.mem",
        "B1.npy": "B1.mem",
        "W2.npy": "W2.mem",
        "B2.npy": "B2.mem",
    }
    
    for npy_f, mem_f in weight_files.items():
        convert_npy_to_mem(npy_f, mem_f, WEIGHT_SCALE_FACTOR, output_format='hex')

    # --- L3 가중치/편향 변환 (별도 처리 및 검증) ---
    print("\n--- 🕵️ L3 (j=1 'alexa') 가중치 검증 ---")
    
    # W3 변환
    w3_float = convert_npy_to_mem("W3.npy", "W3.mem", WEIGHT_SCALE_FACTOR, output_format='hex')
    if w3_float is not None:
        # j=1 ('alexa') 가중치 (W3[1])의 일부(앞 5개)와 통계 출력
        alexa_w_weights = w3_float[1] # Shape (128,)
        print(f"W3[1] (alexa) float (일부): {alexa_w_weights[:5]}")
        print(f"  ... W3[1] Min: {np.min(alexa_w_weights):.4f}, Max: {np.max(alexa_w_weights):.4f}, Mean: {np.mean(alexa_w_weights):.4f}")
        if np.all(alexa_w_weights == 0):
            print("🚨🚨🚨 치명적 오류: W3[1] ('alexa') 가중치가 모두 0입니다!")

    # B3 변환
    b3_float = convert_npy_to_mem("B3.npy", "B3.mem", WEIGHT_SCALE_FACTOR, output_format='hex')
    if b3_float is not None:
        # j=1 ('alexa') 편향 (B3[1]) 값 출력
        alexa_b_weight = b3_float[1]
        print(f"B3[1] (alexa) float: {alexa_b_weight:.4f}")
        if alexa_b_weight == 0:
            print("🚨🚨🚨 경고: B3[1] ('alexa') 편향이 0입니다!")
            

    # --- LIF 파라미터 처리 ---
    try:
        lif_params_path = os.path.join(INPUT_DIR, "LIF_params.npy")
        lif_params = np.load(lif_params_path, allow_pickle=True).item()
        
        # QAT 코드에서 양자화된 float 값을 가져옴
        threshold_float = lif_params['THRESHOLD_VAL'] # 0.5
        
        # 이 float 값을 Q5.11 정수로 변환 (0.5 * 2048 = 1024)
        threshold_int = float_to_fixed_point_int(threshold_float, THRESH_SCALE_FACTOR)
        
        thresh_mem_path = os.path.join(OUTPUT_DIR, "THRESHOLD.mem")
        print(f"\nLIF 임계값 저장 중 -> {thresh_mem_path}")
        with open(thresh_mem_path, 'w') as f:
            hex_str = f'{(threshold_int & 0xFFFF):04X}' # 0400
            f.write(f"{hex_str}\n")
            
        print("\n✅ 모든 변환 완료!")
        
    except FileNotFoundError:
        print(f"🚨 경고: 'LIF_params.npy' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"🚨 LIF 파라미터 처리 중 오류: {e}")