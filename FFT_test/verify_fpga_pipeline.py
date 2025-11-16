import numpy as np
from scipy.io import wavfile
from scipy import signal
import serial
import struct
import time
import os
import sys

# -----------------------------------------------
# [설정] Verilog 파라미터 및 파일 경로
# -----------------------------------------------
# --- Verilog 파라미터 (data_prep.v, frame_window_controller.v 등)
TARGET_SAMPLE_RATE = 8000   # Fs = 8000 [cite: 72]
N_SAMPLES = 29280           # N_SAMPLES = 29280 [cite: 73]
N_FFT = 256                 # N_FFT = 256 [cite: 42, 122]
FRAME_SIZE = 256            # FRAME_SIZE = 256 [cite: 122]
N_FFT_BINS = 129            # 256/2 + 1 [cite: 73]
WAIT_CYCLES_2SEC = 1.0      # S_WAIT_2SEC [cite: 75, 102]

# --- FPGA 연결 설정 (uart_rx.v, uart_tx.v)
SERIAL_PORT = "COM4"        # FPGA가 연결된 COM 포트
BAUD_RATE = 230400          # BAUD = 230400 [cite: 19, 55]

# --- 파일 경로 설정
# [입력 1] 변환할 원본 WAV 파일 (SNN 학습에 사용한 파일)
INPUT_WAV_FILE = "alexa_sample2.wav" 
# [입력 2] Verilog가 읽는 Hamming Window ROM
HAMMING_MEM_FILE = "hamming_q7_9.mem" 
# [cite: 53]
# [중간 파일] 1번을 8비트로 변환하여 저장 (FPGA 전송용)
INPUT_AUDIO_FILE = "input_audio.bin"
# --- 비교 설정 
# 고정소수점 반올림 오차 허용 범위
TOLERANCE = 2
# -----------------------------------------------


# -----------------------------------------------
# 헬퍼 함수
# -----------------------------------------------
def to_q16_signed(value):
    """Python의 정수/실수를 Verilog의 16비트 부호 있는 정수로 변환"""
    value = int(round(value))
    if value > 32767: value = 32767
    elif value < -32768: value = -32768
    return np.int16(value)

def load_hamming_rom(mem_file):
    """Verilog의 $readmemh 파일을 읽어서 Q7.9 정수 배열로 로드 [cite: 126]"""
    coeffs = []
    try:
        with open(mem_file, 'r') as f:
            for line in f:
                hex_val = line.strip()
                if hex_val:
                    uint_val = int(hex_val, 16)
                    coeffs.append(to_q16_signed(uint_val))
    except FileNotFoundError:
        print(f"[에러] Hamming window 파일 '{mem_file}'을 찾을 수 없습니다. ")
        return None
    
    if len(coeffs) != FRAME_SIZE:
        print(f"[경고] {mem_file}에서 {len(coeffs)}개만 로드됨. {FRAME_SIZE}개가 필요함. [cite: 122, 125]")
    return np.array(coeffs, dtype=np.int16)

# -----------------------------------------------
# 기능 1: WAV -> BIN 변환
# -----------------------------------------------
def convert_wav_to_bin():
    """WAV 파일을 FPGA 전송용 8-bit 바이너리 파일로 변환"""
    print(f"--- 1. WAV -> BIN 변환 작업 시작 ---")
    print(f"'{INPUT_WAV_FILE}' 파일을 로드 중...")

    if not os.path.exists(INPUT_WAV_FILE):
        print(f"[에러] 입력 파일 '{INPUT_WAV_FILE}'을 찾을 수 없습니다.")
        return False

    try:
        sample_rate, audio_data = wavfile.read(INPUT_WAV_FILE)
        print(f"로드 성공. 원본 샘플레이트: {sample_rate} Hz, 길이: {len(audio_data)} 샘플")
    except Exception as e:
        print(f"[에러] WAV 파일 로드 실패: {e}")
        return False

    if audio_data.ndim > 1:
        print("스테레오 감지. 모노로 변환 중...")
        audio_data = audio_data.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"샘플레이트 변환 중: {sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz")
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)

    # 16비트 Signed (-32768~32767) -> 8비트 Unsigned (0~255) 변환
    # Verilog의 (X-128) 로직 에 맞추기 위함
    print("16-bit Signed -> 8-bit Unsigned 변환 중...")
    audio_data_8bit = (audio_data / 256.0) + 128.0
    audio_data_8bit = np.clip(audio_data_8bit, 0, 255)
    audio_data_8bit = audio_data_8bit.astype(np.uint8)

    current_length = len(audio_data_8bit)
    if current_length < N_SAMPLES:
        print(f"샘플 길이가 {N_SAMPLES}보다 짧습니다. 128(무음)으로 패딩 중... [cite: 73]")
        pad_width = N_SAMPLES - current_length
        final_data = np.pad(audio_data_8bit, (0, pad_width), 'constant', constant_values=128)
    elif current_length > N_SAMPLES:
        print(f"샘플 길이가 {N_SAMPLES}보다 깁니다. 자르는 중... [cite: 73]")
        final_data = audio_data_8bit[:N_SAMPLES]
    else:
        final_data = audio_data_8bit

    try:
        final_data.tofile(INPUT_AUDIO_FILE)
        print(f"\n[성공] '{INPUT_AUDIO_FILE}' 파일 생성 완료! ({len(final_data)} 바이트)")
        return True
    except Exception as e:
        print(f"\n[에러] 파일 저장 실패: {e}")
        return False

# -----------------------------------------------
# 기능 2: Python 레퍼런스 모델 (Golden Model)
# -----------------------------------------------
def generate_reference_model():
    """BIN 파일과 Verilog 로직을 기반으로 '정답' Magnitude 리스트를 생성"""
    print(f"--- 2. Python 레퍼런스 모델 실행 ---")
    
    # (1) 8비트 오디오 샘플 로드 (첫 프레임만)
    try:
        audio_samples_8bit = np.fromfile(INPUT_AUDIO_FILE, dtype=np.uint8, count=FRAME_SIZE)
        if len(audio_samples_8bit) < FRAME_SIZE:
             print(f"[에러] '{INPUT_AUDIO_FILE}' 파일에 샘플이 {len(audio_samples_8bit)}개뿐입니다. [cite: 122]")
             return None
    except FileNotFoundError:
        print(f"[에러] '{INPUT_AUDIO_FILE}' 파일을 찾을 수 없습니다. (1번 메뉴 먼저 실행)")
        return None

    # (2) Hamming Window ROM 로드
    window_coeffs_q7_9 = load_hamming_rom(HAMMING_MEM_FILE)
    if window_coeffs_q7_9 is None:
        return None

    print("Verilog 로직 복제 (Windowing, Q-Format) 중...")
    windowed_samples = []

    # (3) frame_window_controller의 'always' 블록 복제 [cite: 129-143]
    for i in range(FRAME_SIZE):
        ram_douta = audio_samples_8bit[i] # 8-bit 
        
        # Verilog: q7_9_sample <= (ram_douta << 7) - 16'h8000; 
        q_sample = (int(ram_douta) - 128) << 7 
        q_sample = to_q16_signed(q_sample)

        q7_9_window_coeff = window_coeffs_q7_9[i] # [cite: 137]

        # Q-format 곱셈 (Q8.7 * Q7.9 = Q15.16)
        product_q15_16 = np.int64(q_sample) * np.int64(q7_9_window_coeff) # [cite: 138]
        
        # Quantize (Q15.16 -> Q15.7)
        # Verilog: (product_q14_18 + 9'h100) >> 9; [cite: 140] (주석은 Q14.18이지만 실제론 Q15.16)
        rounding_val = (1 << 8) # 256
        quantized_sample = (product_q15_16 + rounding_val) >> 9
        
        final_sample = to_q16_signed(quantized_sample)
        windowed_samples.append(final_sample)

    # (4) FFT IP Core (256 Point) 복제
    fft_input = np.array(windowed_samples, dtype=np.int16)
    # FFT IP Core가 "Scaled (Divide by N)"로 설정되었다고 가정 (N=256)
    fft_result_complex = np.fft.fft(fft_input, N_FFT) / N_FFT # [cite: 48]

    print("FFT 및 Magnitude 계산 중...")

    # (5) Magnitude Calculator 복제 [cite: 1-17]
    reference_magnitudes = []
    for i in range(N_FFT_BINS): # 0~128 [cite: 73]
        complex_val = fft_result_complex[i]
        
        R_in = to_q16_signed(complex_val.real)
        I_in = to_q16_signed(complex_val.imag)

        abs_R = np.abs(R_in) # [cite: 5]
        abs_I = np.abs(I_in) # [cite: 5]
        max_val = max(abs_R, abs_I) # [cite: 6]
        min_val = min(abs_R, abs_I) # [cite: 7]
        half_min_val = min_val >> 1 # [cite: 9]

        sum_result = np.int32(max_val) + np.int32(half_min_val) # [cite: 11]

        if sum_result > 32767: # [cite: 15]
            mag_out = 32767    # [cite: 15]
        elif sum_result < -32768:
            mag_out = -32768
        else:
            mag_out = np.int16(sum_result) # [cite: 16]

        reference_magnitudes.append(mag_out)
    
    print(f"[성공] Python 레퍼런스 Magnitude {len(reference_magnitudes)}개 생성 완료.")
    return reference_magnitudes

# -----------------------------------------------
# 기능 3: FPGA 테스트 실행 (데이터 전송 및 수신)
# -----------------------------------------------
def run_fpga_test():
    """FPGA로 8-bit 오디오를 전송하고, 16-bit Magnitude 결과를 수신"""
    print(f"--- 3. FPGA 테스트 실행 ---")
    
    # (1) 전송할 오디오 파일 로드
    try:
        with open(INPUT_AUDIO_FILE, 'rb') as f:
            audio_data = f.read()
        if len(audio_data) != N_SAMPLES:
            print(f"[경고] '{INPUT_AUDIO_FILE}' 크기({len(audio_data)})가 N_SAMPLES({N_SAMPLES})와 다릅니다.")
    except FileNotFoundError:
        print(f"[에러] '{INPUT_AUDIO_FILE}' 파일을 찾을 수 없습니다. (1번 메뉴 먼저 실행)")
        return None
    except Exception as e:
        print(f"[에러] 오디오 파일 로드 실패: {e}")
        return None

    print(f"'{INPUT_AUDIO_FILE}' 로드 완료 ({len(audio_data)} 바이트).")

    # (2) 시리얼 포트 연결
    total_bytes_to_read = N_FFT_BINS * 2 
    fpga_magnitudes = []
    
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10) as ser:
        print(f"{SERIAL_PORT}에서 {BAUD_RATE} bps로 연결 시도 중... [cite: 19, 55]")
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10) as ser:
            print(f"연결 성공.")
            
            # (3) FPGA로 오디오 데이터 전송 (S_RECORD 루프) [cite: 99-102]
            print(f"FPGA로 {N_SAMPLES} 바이트 전송 시작...")
            start_time = time.time()
            bytes_sent = ser.write(audio_data)
            ser.flush() 
            end_time = time.time()
            print(f"전송 완료. {bytes_sent} 바이트 전송 (소요 시간: {end_time - start_time:.2f}초)")
            
            if bytes_sent != N_SAMPLES:
                print(f"[에러] 전송 실패! {bytes_sent}/{N_SAMPLES} 바이트만 전송됨.")
                return None

            # (4) FPGA의 2초 대기 및 FFT 처리 시간 기다리기 [cite: 102-105]
            wait_time = WAIT_CYCLES_2SEC + 0.5 # 여유 시간 0.5초
            print(f"FPGA 2초 대기 및 FFT 처리 중... (총 {wait_time:.1f}초 대기)")
            time.sleep(wait_time)

            # (5) FPGA로부터 Magnitude 데이터 수신 (S_MAG_TX, S_TX_MSB) [cite: 108-115]
            print(f"FPGA로부터 {total_bytes_to_read} 바이트의 Magnitude 데이터 수신 시작...")
            raw_bytes = ser.read(total_bytes_to_read)

            if len(raw_bytes) != total_bytes_to_read:
                print(f"[에러] 데이터 수신 실패. 예상: {total_bytes_to_read}, 수신: {len(raw_bytes)} 바이트")
                return None
            
            print("데이터 수신 완료. 16비트로 변환 중...")
            # LSB, MSB 순서로 16비트 정수 변환 (Little-Endian)
            for i in range(0, total_bytes_to_read, 2):
                byte_pair = raw_bytes[i:i+2]
                value = struct.unpack('<h', byte_pair)[0] # LSB 먼저 [cite: 109, 113]
                fpga_magnitudes.append(value)

        print(f"[성공] FPGA Magnitude {len(fpga_magnitudes)}개 수신 완료.")
        return fpga_magnitudes

    except serial.SerialException as e:
        print(f"[치명적 에러] 시리얼 포트 오류: {e}")
        print("1. COM 포트 번호 확인, 2. FPGA 전원 및 연결 확인")
        return None
    except Exception as e:
        print(f"[일반 에러] {e}")
        return None

# -----------------------------------------------
# 기능 4: 결과 비교
# -----------------------------------------------
def compare_results(ref_data, fpga_data):
    """메모리에 있는 두 리스트(레퍼런스 vs FPGA)를 비교"""
    print(f"--- 4. 최종 결과 비교 ---")
    if ref_data is None or fpga_data is None:
        print("[에러] 비교할 데이터가 없습니다. (2, 3번 기능 실패)")
        return

    if len(fpga_data) != len(ref_data):
        print(f"[경고] 데이터 개수 불일치! (FPGA: {len(fpga_data)}, Ref: {len(ref_data)})")
        return

    print(f"오차 허용 범위: +/- {TOLERANCE}")
    print("Bin # | FPGA 출력 | Python 레퍼런스 | 차이 | 결과")
    print("-----------------------------------------------------")
    
    mismatch_count = 0
    for i in range(len(ref_data)):
        fpga_val = fpga_data[i]
        ref_val = ref_data[i]
        diff = int(fpga_val) - int(ref_val)
        
        if abs(diff) <= TOLERANCE:
            status = "일치 (PASS)"
        else:
            status = f"불일치 (FAIL) <---"
            mismatch_count += 1
        
        # 100개까지만 상세 출력 (너무 길어지는 것 방지)
        if i < 100 or status.startswith("불일치"):
             print(f"{i:<5} | {fpga_val:<10} | {ref_val:<16} | {diff:<4} | {status}")
        elif i == 100:
            print("... (일치하는 중간 결과는 생략) ...")

    print("-----------------------------------------------------")
    if mismatch_count == 0:
        print(f"\n[최종 결론: 완벽히 일치 (PASS, 오차범위 {TOLERANCE})]")
        print("축하합니다! FPGA 로직이 Python 모델과 정확히 일치합니다.")
    else:
        print(f"\n[최종 결론: {mismatch_count}개 불일치 (FAIL)]")
        print("불일치 지점을 확인하고 Verilog 로직 또는 Python 모델을 다시 검토하세요.")

# -----------------------------------------------
# 메인 실행 함수
# -----------------------------------------------
def main():
    print("=========================================")
    print(" FPGA - SNN 파이프라인 검증 스크립트")
    print("=========================================")
    print(f"  WAV 파일: {INPUT_WAV_FILE}")
    print(f"  BIN 파일: {INPUT_AUDIO_FILE}")
    print(f"  COM 포트: {SERIAL_PORT} @ {BAUD_RATE} Bps")
    print("-----------------------------------------")
    print("수행할 작업을 선택하세요:")
    print("  1. WAV -> BIN 변환 (오디오 파일 준비)")
    print("  2. 전체 검증 실행 (BIN 파일 -> FPGA -> 결과 비교)")
    print("  Q. 종료")
    
    while True:
        choice = input("선택 (1, 2, Q): ").strip().upper()
        
        if choice == '1':
            convert_wav_to_bin()
            break
        
        elif choice == '2':
            if not os.path.exists(INPUT_AUDIO_FILE):
                print(f"[경고] '{INPUT_AUDIO_FILE}'이 없습니다. 먼저 1번을 실행해야 합니다.")
                if input("지금 1번(WAV->BIN 변환)을 실행하시겠습니까? (y/n): ").strip().lower() == 'y':
                    if not convert_wav_to_bin():
                        print("[에러] BIN 파일 생성 실패. 검증 중단.")
                        break
                else:
                    print("검증 중단.")
                    break
            
            # 2. 레퍼런스 모델 실행
            ref_mags = generate_reference_model()
            
            # 3. FPGA 테스트 실행
            fpga_mags = run_fpga_test()
            
            # 4. 결과 비교
            compare_results(ref_mags, fpga_mags)
            break
            
        elif choice == 'Q':
            print("종료합니다.")
            break
        
        else:
            print("잘못된 입력입니다. 1, 2, Q 중에서 선택하세요.")

if __name__ == "__main__":
    # 라이브러리 설치 확인
    try:
        import numpy
        import scipy
        import serial
    except ImportError as e:
        print(f"[치명적 에러] 필수 라이브러리가 없습니다: {e.name}")
        print(f"다음 명령어로 설치하세요: pip install numpy scipy pyserial")
        sys.exit(1)
        
    main()