import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import glob # 파일 목록 검색을 위해 glob 라이브러리 추가

# ----------------------------------------------------
# FPGA 최적화 설정 변수 (Nexys A7 친화적)
# ----------------------------------------------------
SAMPLE_RATE = 8000       # 8kHz
FRAME_SIZE = 256         # 32ms (N_FFT와 동일)
HOP_LENGTH = 80          # 10ms (실시간 처리 주기)
N_MELS = 20              # 멜 필터 뱅크 개수
N_FFT = 256              # FFT 크기
NUM_TIMESTEPS = 10       # 10ms 프레임당 SNN 시뮬레이션 스텝 수 (1ms/step)

wav_folder_path = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed"
spike_output_path = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed/spike"

# ----------------------------------------------------
# A. Mel Filterbank 특징 추출 함수 (기존 코드)
# ----------------------------------------------------
def extract_optimized_mel_filterbank(audio_data):
    """
    Nexys A7 친화적인 파라미터로 Mel Filterbank 특징 추출
    """
    
    # STFT
    stft_result = librosa.stft(
        y=audio_data,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=FRAME_SIZE,
        center=False 
    )
    
    # 크기 스펙트럼
    magnitude_spectrum = np.abs(stft_result)
    
    # 멜 필터 뱅크 행렬 생성 및 적용
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS
    )
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrum)
    
    # 로그 스케일 변환
    log_mel_features = np.log(mel_spectrogram + 1e-6)
    
    return log_mel_features.T # (N_frames, N_MELS)

# ----------------------------------------------------
# B. Poisson Rate Coding 함수 (새로운 코드)
# ----------------------------------------------------
def poisson_encode_for_snn(log_mel_features, num_timesteps=NUM_TIMESTEPS):
    np.random.seed(42) # 원하는 임의의 정수를 사용하세요. 

    # 1. 정규화 (Normalization)
    min_val = np.min(log_mel_features)
    max_val = np.max(log_mel_features)
    range_val = max_val - min_val
    
    if range_val < 1e-6:
        probabilities = np.full_like(log_mel_features, 0.5) 
    else:
        # 특징 값을 [0, 1] 범위의 확률로 변환
        probabilities = (log_mel_features - min_val) / range_val
    
    # 2. 스파이크 열 생성
    num_frames, n_mels = log_mel_features.shape
    total_timesteps = num_frames * num_timesteps
    
    # 🚨 [치명적 수정] dtype을 np.int8에서 np.float32로 변경
    spike_train = np.zeros((total_timesteps, n_mels), dtype=np.float32) 
    
    for i in range(num_frames):
        P = probabilities[i, :]
        
        for t in range(num_timesteps):
            idx = i * num_timesteps + t
            # P 확률로 스파이크 생성
            # 🚨 astype도 np.float32로 변경
            spike_train[idx, :] = (np.random.rand(n_mels) < P).astype(np.float32)
            
    return spike_train

# ----------------------------------------------------
# 메인 실행 블록
# ----------------------------------------------------
if __name__=="__main__":

    # 1. 파일 목록 생성 및 출력 폴더 준비
    # wav_folder_path 내의 모든 WAV 파일 경로를 가져옵니다.
    all_wav_paths = glob.glob(os.path.join(wav_folder_path, "*.wav"))
    
    if not all_wav_paths:
        print(f"경고: {wav_folder_path} 경로에서 WAV 파일을 찾을 수 없습니다.")
        exit()

    os.makedirs(spike_output_path, exist_ok=True)
    print(f"총 {len(all_wav_paths)}개의 WAV 파일을 처리합니다. NPY는 {spike_output_path}에 저장됩니다.")

    # 2. 파일 처리 루프 (파일명에 의존하지 않음)
    for i, wav_path in enumerate(all_wav_paths):
        # 파일명 추출
        file_name = os.path.basename(wav_path)
        base_name, _ = os.path.splitext(file_name)

        # ----------------------------------------------------
        # 3. 오디오 로드 및 특징 추출
        # ----------------------------------------------------
        try:
            # 파일을 로드합니다. (이전 단계에서 8kHz로 전처리했다고 가정)
            audio_data, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
            
            # Mel Filterbank 특징 추출
            features = extract_optimized_mel_filterbank(audio_data)

            # 4. Spike Train 인코딩
            spike_input = poisson_encode_for_snn(features, num_timesteps=NUM_TIMESTEPS)
            
            # 5. Spike Train 저장
            # 파일명에 dtype 정보를 넣어 np.int8로 생성된 이전 파일과 구분하는 것이 좋습니다.
            save_filename = f"spike_input_f32_{i+1}.npy"
            np.save(os.path.join(spike_output_path, save_filename), spike_input)

            # 6. 결과 출력
            if (i + 1) % 50 == 0 or (i + 1) == len(all_wav_paths):
                print(f"--- {i+1}/{len(all_wav_paths)} 처리 완료 ---")
                print(f"원본: {file_name} -> Spike Train 형태: {spike_input.shape} 저장됨.")


        except Exception as e:
            print(f"파일 처리 오류 발생 ({file_name}): {e}")
            continue

    print(f"\n--- 전체 Negative Spike Train 변환 완료 ---")
    
    # 시각화 부분은 일괄 처리에 방해가 되므로 주석 처리하거나, 필요 시 개별 파일에 대해 실행하세요.
    # (일괄 처리 시 수백개의 창이 열리는 것을 방지하기 위함)