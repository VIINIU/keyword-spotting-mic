import numpy as np

# 1. 설정
T_MAX = 3000
N_MELS = 20
NPY_SPIKE_FILE_PATH = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed/spike_16bit_regenerated/spike_input_16bit_1.npy" # 🚨 검증할 npy 파일 1개 지정
TB_SPIKE_TXT_FILE = "./spike_stimulus_1.txt" # 🚨 테스트벤치가 읽을 파일

# 2. .npy 로드
try:
    spike_data = np.load(NPY_SPIKE_FILE_PATH)
except Exception as e:
    print(f"오류: {NPY_SPIKE_FILE_PATH} 로드 실패 - {e}")
    exit()

# 3. 패딩/절삭 (QAT 코드와 동일하게)
if spike_data.shape[0] > T_MAX:
    spike_data = spike_data[:T_MAX, :]
elif spike_data.shape[0] < T_MAX:
    padding = np.zeros((T_MAX - spike_data.shape[0], N_MELS), dtype=np.float32)
    spike_data = np.vstack([spike_data, padding])

print(f"변환 시작: {spike_data.shape} -> {TB_SPIKE_TXT_FILE}")

# 4. .txt 파일로 변환 (16진수)
with open(TB_SPIKE_TXT_FILE, 'w') as f:
    for t in range(T_MAX):
        # 20비트 벡터 [0, 1, 0, ..., 1]
        spike_vector_bits = spike_data[t, :]
        
        # 2진수 문자열로 변환 "010...1"
        bin_str = "".join(str(int(bit)) for bit in spike_vector_bits)
        
        # 20비트 2진수 문자열을 -> 16진수 정수로 변환
        hex_val = int(bin_str, 2)
        
        # 16진수 문자열로 포맷팅 (20비트 = 5 헥스 문자)
        hex_str = f"{hex_val:05X}" # 예: 00000, 00001, 80000, FFFFF
        
        f.write(f"{hex_str}\n")

print(f"✅ 변환 완료: {TB_SPIKE_TXT_FILE} 생성 (3000 라인)")