import numpy as np
import librosa
import os
import glob

# ----------------------------------------------------
# 📜 A. 기본 설정 및 경로
# ----------------------------------------------------
# FPGA 최적화 설정 변수
SAMPLE_RATE = 8000
FRAME_SIZE = 256
HOP_LENGTH = 80
N_MELS = 20
N_FFT = 256
NUM_TIMESTEPS = 30
GAIN_FACTOR = 0.3  # <-- [교수의 조언] SNN 학습 결과에 따라 이 값을 0.1~0.3 사이로 조절하게

# [경로 수정] - 자네의 환경에 맞게 경로를 설정하게
# positive command paths
wav_path_root = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed"
spike_path_root = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed/spike_16bit_regenerated"
# negative command paths
# wav_path_root = "C:/Users/11e26/Desktop/internship/source/clear_negative_command"
# spike_path_root = "C:/Users/11e26/Desktop/internship/source/clear_negative_command/spike_16bit_regenerated"

# [!] 주의: 아래의 GLOBAL 값들은 1단계(Pass 1)에서 계산된
# '실제' 값으로 덮어쓰일 예정이므로, 더 이상 사용되지 않음!
# GLOBAL_MIN_LOG_MEL = -14.0 # (삭제됨)
# GLOBAL_MAX_LOG_MEL = 0.0   # (삭제됨)

# ----------------------------------------------------
# 🔊 B. Mel Filterbank 특징 추출 함수 (이전과 동일)
# ----------------------------------------------------
def extract_optimized_mel_filterbank(audio_data_float):
    """
    Mel Filterbank 특징 추출. 16bit 양자화 후 float으로 변환된 데이터를 입력으로 받습니다.
    """
    stft_result = librosa.stft(
        y=audio_data_float,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=FRAME_SIZE,
        center=False
    )
    magnitude_spectrum = np.abs(stft_result)
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrum)
    log_mel_features = np.log(mel_spectrogram + 1e-6)

    return magnitude_spectrum, log_mel_features.T # (Magnitude, Log Mel Features)

# ----------------------------------------------------
# ⚡ C. Poisson Rate Coding 함수 (!!핵심 수정!!)
# ----------------------------------------------------
def poisson_encode_for_snn(log_mel_features,
                           actual_min_log,  # <-- [수정] 1단계에서 찾은 '실제' 최솟값
                           actual_max_log,  # <-- [수정] 1단계에서 찾은 '실제' 최댓값
                           num_timesteps=NUM_TIMESTEPS,
                           gain_factor=GAIN_FACTOR):
    """
    [수정]
    데이터셋 전체의 '실제' Min/Max 값을 인자로 받아 정규화를 수행합니다.
    """
    np.random.seed(42)

    # 1. 실제 Min/Max 값으로 클리핑
    clipped_features = np.clip(log_mel_features, actual_min_log, actual_max_log)

    # 2. 실제 Range로 정규화
    actual_range = actual_max_log - actual_min_log
    if actual_range < 1e-6:
        probabilities_0_to_1 = np.full_like(clipped_features, 0.0)
    else:
        # (값 - 최소값) / (최대값 - 최소값) => 0~1 사이로 정규화
        probabilities_0_to_1 = (clipped_features - actual_min_log) / actual_range

    # 3. 희소성(Sparsity)을 위한 확률 스케일링 (이전과 동일)
    P_scaled = probabilities_0_to_1 * gain_factor

    num_frames, n_mels = log_mel_features.shape
    total_timesteps = num_frames * num_timesteps
    spike_train = np.zeros((total_timesteps, n_mels), dtype=np.float32)

    for i in range(num_frames):
        P = P_scaled[i, :]
        for t in range(num_timesteps):
            idx = i * num_timesteps + t
            spike_train[idx, :] = (np.random.rand(n_mels) < P).astype(np.float32)

    return spike_train

# ----------------------------------------------------
# 🔍 D. [신규] 1단계(Pass 1): 동적 범위 분석 함수
# ----------------------------------------------------
def run_pass_1_analysis(wav_dir):
    """
    데이터셋 전체를 스캔하여 실제 Min/Max 동적 범위를 찾습니다.
    (스파이크 파일은 생성하지 않습니다.)
    """
    print("--- 1단계 (Pass 1): 전체 데이터셋 동적 범위 분석 시작 ---")
    max_magnitude = 0.0
    min_log_mel = np.inf
    max_log_mel = -np.inf

    all_wav_paths = glob.glob(os.path.join(wav_dir, "*.wav"))
    if not all_wav_paths:
        print(f"[경고] {wav_dir} 에서 WAV 파일을 찾을 수 없습니다!")
        return min_log_mel, max_log_mel, max_magnitude
        
    print(f"총 {len(all_wav_paths)}개 파일 분석 중...")

    for i, wav_path in enumerate(all_wav_paths):
        audio_data_float, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        # 16bit Signed Int (int16) 양자화 시뮬레이션
        MAX_INT16 = 32767.0
        audio_data_int16 = (audio_data_float * MAX_INT16).astype(np.int16)
        audio_data_for_stft = (audio_data_int16.astype(np.float32) / MAX_INT16)

        # 특징 추출
        magnitude_spectrum, features = extract_optimized_mel_filterbank(audio_data_for_stft)

        # 동적 범위 업데이트
        max_magnitude = max(max_magnitude, np.max(magnitude_spectrum))
        min_log_mel = min(min_log_mel, np.min(features))
        max_log_mel = max(max_log_mel, np.max(features))

        if (i + 1) % 100 == 0 or (i + 1) == len(all_wav_paths):
            print(f"  ... {i+1} / {len(all_wav_paths)}개 파일 분석 완료")

    print("✅ 1단계 (Pass 1) 분석 완료.")
    return min_log_mel, max_log_mel, max_magnitude

# ----------------------------------------------------
# 💾 E. [신규] 2단계(Pass 2): 스파이크 트레인 생성 함수
# ----------------------------------------------------
def run_pass_2_generation(wav_dir, spike_dir, actual_min_log, actual_max_log):
    """
    1단계에서 찾은 Min/Max 값을 기준으로 스파이크 트레인을 생성하고 저장합니다.
    """
    print("\n--- 2단계 (Pass 2): 스파이크 트레인 생성 시작 ---")
    print(f"✅ 적용될 실제 Log-Mel 범위: [{actual_min_log:.4f}, {actual_max_log:.4f}]")

    os.makedirs(spike_dir, exist_ok=True)
    all_wav_paths = glob.glob(os.path.join(wav_dir, "*.wav"))
    print(f"총 {len(all_wav_paths)}개 파일 변환 시작...")

    for i, wav_path in enumerate(all_wav_paths):
        audio_data_float, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        # 16bit 양자화 시뮬레이션
        MAX_INT16 = 32767.0
        audio_data_int16 = (audio_data_float * MAX_INT16).astype(np.int16)
        audio_data_for_stft = (audio_data_int16.astype(np.float32) / MAX_INT16)

        # 특징 추출 (여기서는 Magnitude 스펙트럼은 필요 없음)
        _, features = extract_optimized_mel_filterbank(audio_data_for_stft)

        # Spike Train 인코딩 (!!수정된 함수 호출!!)
        spike_input = poisson_encode_for_snn(features,
                                           actual_min_log,  # <-- 1단계 결과 전달
                                           actual_max_log)  # <-- 1단계 결과 전달

        # 저장
        save_filename = f"spike_input_16bit_{i+1}.npy"
        np.save(os.path.join(spike_dir, save_filename), spike_input)

    print("✅ 2단계 (Pass 2) 스파이크 트레인 생성 완료.")

# ----------------------------------------------------
# 🚀 F. Main 실행 블록 (수정본 - 설명 주석 추가)
# ----------------------------------------------------
if __name__ == "__main__":

    # ✅ [1단계] : "Pass 1" 함수가 여기서 '먼저' 실행된다네.
    # 이 함수가 끝나면 'final_...' 변수들에
    # 데이터셋 전체의 실제 Min/Max 값이 저장되지.
    print(">>> 1단계(분석)를 시작합니다...")
    final_min_log, final_max_log, final_max_mag = run_pass_1_analysis(wav_path_root)

    # ✅ [중간 점검] : "Pass 1"이 잘 되었는지 확인
    # 만약 1단계에서 파일을 못 찾아서 값이 무한대(inf)로 남아있다면,
    # 2단계를 실행하지 않고 여기서 멈춘다네.
    if np.isinf(final_min_log) or np.isinf(final_max_log):
        print("\n[!!!] 오류: 1단계 분석에서 유효한 Log-Mel 범위를 찾지 못했습니다.")
        print("WAV 파일 경로(wav_path_root)가 올바른지 확인하게.")
    
    else:
        # ✅ [2단계] : "Pass 1"이 성공했으므로, "Pass 2" 함수가 '바로 이어서' 실행된다네.
        # [중요] 1단계의 결과물인 final_min_log, final_max_log 값을
        # 2단계 함수의 인자(argument)로 '그대로 전달'하지.
        #
        # 즉, 자네가 수동으로 값을 복사/붙여넣기 할 필요 없이
        # 스크립트가 '알아서' 1단계 결과를 2단계에서 사용하네.
        print("\n>>> 2단계(생성)를 시작합니다...")
        run_pass_2_generation(wav_path_root, 
                              spike_path_root, 
                              final_min_log,  # <-- 1단계 결과가 자동으로 전달됨
                              final_max_log)  # <-- 1단계 결과가 자동으로 전달됨

        # ✅ [3단계] : "Pass 2"까지 모두 완료된 후, 최종 분석 결과를 화면에 출력한다네.
        # 이 결과는 자네가 나중에 'Verilog' 코드를 짤 때 참고하라는 걸세.
        print("\n--- 16bit 양자화 기준 Fixed Point 동적 범위 분석 결과 (Pass 1) ---")
        print(f"1. [실제] Magnitude Spectrum Max: {final_max_mag:.4f}")
        print(f"2. [실제] Log Mel Feature Range: [{final_min_log:.4f}, {final_max_log:.4f}]")

        # ... (이하 Fixed Point Qm.n 포맷 결정 예시 부분은 동일) ...
        
        TOTAL_BITS = 16
        
        # [학생이 직접 판단할 부분]
        # 위 1, 2번 '실제' 값을 보고 Verilog에 쓸 m_mag, m_log 값을 결정하게.
        m_mag = 7
        n_mag = TOTAL_BITS - m_mag
        m_log = 9 
        n_log = TOTAL_BITS - m_log

        print("\n--- Qm.n 포맷 최종 결정 (Verilog 설계 참고용 '예시') ---")
        print(f"1. STFT Magnitude, Mel Spectrogram: Q{m_mag}.{n_mag} (추정)")
        print(f"   - 표현 가능 범위: +/-{2**(m_mag-1) - (2**(-n_mag)):.4f} (약 +/-{2**(m_mag-1):.0f})")
        print(f"2. Log Mel Features: Q{m_log}.{n_log} (추정)")
        print(f"   - 표현 가능 범위: +/-{2**(m_log-1) - (2**(-n_log)):.4f} (약 +/-{2**(m_log-1):.0f})")
        print(f"   - 소수부 정밀도: {2**(-n_log):.8f}")