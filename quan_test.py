import numpy as np
import librosa
import glob
import os

# FPGA 최적화 설정 변수 (이전 코드와 동일)
SAMPLE_RATE = 8000       
FRAME_SIZE = 256         
HOP_LENGTH = 80          
N_MELS = 20              
N_FFT = 256              

# 데이터셋 경로 (실제 WAV 파일이 있는 경로로 변경하세요)
wav_folder_path = "C:/path/to/your/wav/files" 

# --- 특징 추출 함수 (이전 코드와 동일) ---
def extract_optimized_mel_filterbank(audio_data):
    # ... (함수 내용 생략 - STFT, Magnitude, Mel Basis, Log)
    stft_result = librosa.stft(y=audio_data, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=FRAME_SIZE, center=False)
    magnitude_spectrum = np.abs(stft_result)
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrum)
    log_mel_features = np.log(mel_spectrogram + 1e-6)
    return magnitude_spectrum, log_mel_features

# --- 범위 분석 메인 로직 ---
def analyze_dynamic_range(folder_path):
    all_wav_paths = glob.glob(os.path.join(folder_path, "*.wav"))
    
    if not all_wav_paths:
        print(f"경고: {folder_path} 경로에서 WAV 파일을 찾을 수 없습니다.")
        return

    max_magnitude = 0.0
    min_log_mel = np.inf
    max_log_mel = -np.inf
    
    for wav_path in all_wav_paths:
        # 16bit 오디오를 float로 로드 (정밀도 유지를 위해)
        audio_data, sr = librosa.load(wav_path, sr=SAMPLE_RATE) 
        
        magnitude, log_mel = extract_optimized_mel_filterbank(audio_data)
        
        # 1단계: Magnitude 최댓값 추적
        max_magnitude = max(max_magnitude, np.max(magnitude))
        
        # 2단계: Log Mel 최솟값/최댓값 추적
        min_log_mel = min(min_log_mel, np.min(log_mel))
        max_log_mel = max(max_log_mel, np.max(log_mel))
        
    print("\n--- 동적 범위 분석 결과 ---")
    print(f"1. Magnitude Spectrum Max: {max_magnitude:.4f}")
    print(f"2. Log Mel Feature Range: [{min_log_mel:.4f}, {max_log_mel:.4f}]")
    
    # Q 포맷 가이드라인 제시
    # (예시)
    # Magnitude Max가 1000 이하라면 Q10.6을 사용할 수 있습니다.
    # Log Mel Range가 [-10, 5] 이하라면 Q4.12를 사용할 수 있습니다.

    # SNN 학습 시 Log Mel Feature의 범위가 중요합니다.
    # 이 결과를 기반으로 Qm.n 포맷을 확정합니다.

# analyze_dynamic_range(wav_folder_path) # 실제 경로로 변경 후 실행