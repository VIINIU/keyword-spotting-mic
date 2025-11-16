import numpy as np
import os

# ----------------------------------------------------
# 1. 설정
# ----------------------------------------------------

# 🚨 [중요] [2, 7]을 만들었던 "좋은" NPY 파일 경로
NPY_INPUT_FILE = "C:/Users/11e26/Desktop/internship/source/clear_negative_command/spike_16bit_regenerated/spike_input_16bit_2.npy"

# 🚨 [중요] Verilog 테스트벤치(tb_snn_core.v)가 읽을 새 .txt 파일 경로
# (기존 ...1.txt와 겹치지 않게 'GOOD' 접미사 추가)
TXT_OUTPUT_FILE = "C:/vini_dir/kws_mic/spike_stimulus_GOOD_3_neg.txt"

N_MELS = 20
T_MAX = 3000

# ----------------------------------------------------
# 2. 메인 변환 로직
# ----------------------------------------------------
def convert_npy_to_txt():
    print(f"변환 시작: {NPY_INPUT_FILE} -> {TXT_OUTPUT_FILE}")
    
    try:
        spike_data_np = np.load(NPY_INPUT_FILE)
    except Exception as e:
        print(f"🚨 NPY 파일 로드 실패: {e}")
        return

    # (패딩/절삭 로직)
    if spike_data_np.shape[0] > T_MAX:
        print(f"경고: 원본 {spike_data_np.shape[0]} 스텝을 {T_MAX}로 절삭합니다.")
        spike_data_np = spike_data_np[:T_MAX, :]
    elif spike_data_np.shape[0] < T_MAX:
        print(f"원본 {spike_data_np.shape[0]} 스텝을 {T_MAX}로 패딩합니다.")
        padding = np.zeros((T_MAX - spike_data_np.shape[0], N_MELS), dtype=np.float32)
        spike_data_np = np.vstack([spike_data_np, padding])
        
    # (T_MAX, 20)
    
    with open(TXT_OUTPUT_FILE, 'w') as f:
        for t in range(T_MAX):
            # (20,) shape의 1개 타임스텝 벡터
            time_step_vector = spike_data_np[t, :]
            
            # 1. float (0.0, 1.0) -> int (0, 1)
            # (Verilog [19:0] 순서에 맞게 reversed)
            bin_str = "".join(['1' if x > 0 else '0' for x in reversed(time_step_vector)])
            
            # 2. 20비트 2진수 -> 16진수 5자리
            hex_str = f'{int(bin_str, 2):05X}'
            
            f.write(f"{hex_str}\n")
            
    print(f"✅ 변환 완료! {T_MAX} 라인의 16진수 스파이크가 {TXT_OUTPUT_FILE}에 저장되었습니다.")
    print(f"  (T=0 스파이크 Hex: {int(''.join(['1' if x > 0 else '0' for x in reversed(spike_data_np[0, :])]), 2):05X})")


if __name__ == "__main__":
    convert_npy_to_txt()