import numpy as np
import librosa
import os
import glob
import soundfile as sf # WAV 파일 저장을 위해 soundfile 라이브러리 사용

# pip install librosa soundfile numpy

# ----------------------------------------------------
# A. 설정 변수
# ----------------------------------------------------
SAMPLE_RATE = 8000 # 8kHz
TARGET_RATIO_FACTOR = 4  # 원본 1개당 3개의 증강본 생성 (총 4배)

# Alexa 원본 WAV 폴더 (예시 경로, 실제 경로로 변경 필요)
original_wav_folder = "C:/Users/11e26/Desktop/internship/source/clear_command"
# 증강된 WAV 파일을 저장할 새 폴더
output_wav_folder = "C:/Users/11e26/Desktop/internship/source/clear_command" 

# ----------------------------------------------------
# B. 증강 함수
# ----------------------------------------------------
def augment_audio(y, sr, base_filename, output_dir):
    """오디오 데이터에 Pitch Shift와 Time Stretch를 적용하고 저장합니다."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 원본 저장 (Original)
    sf.write(os.path.join(output_dir, f"{base_filename}_orig.wav"), y, sr)
    
    # 2. 피치 이동 (Pitch Shift): 약간 높은 톤 (0.5 반음)
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
    sf.write(os.path.join(output_dir, f"{base_filename}_pu.wav"), y_pitch_up, sr)

    # 3. 시간 늘이기 (Time Stretch): 약간 느리게 (Rate 0.9)
    y_stretch_slow = librosa.effects.time_stretch(y, rate=0.9)
    # Time Stretch는 길이가 달라지므로, T_MAX 3초에 맞게 자르거나 패딩 필요
    # 여기서는 저장 후 다음 단계(Spike 변환)에서 일괄 처리합니다.
    sf.write(os.path.join(output_dir, f"{base_filename}_sl.wav"), y_stretch_slow, sr)

    # 4. 시간 압축 (Time Stretch): 약간 빠르게 (Rate 1.1)
    y_stretch_fast = librosa.effects.time_stretch(y, rate=1.1)
    sf.write(os.path.join(output_dir, f"{base_filename}_fa.wav"), y_stretch_fast, sr)

    return 4 # 총 4개의 파일 생성

# ----------------------------------------------------
# C. 메인 실행 블록
# ----------------------------------------------------
if __name__=="__main__":
    
    all_wav_paths = glob.glob(os.path.join(original_wav_folder, "*.wav"))
    if not all_wav_paths:
        print(f"오류: {original_wav_folder} 경로에서 WAV 파일을 찾을 수 없습니다.")
        exit()

    print(f"총 {len(all_wav_paths)}개의 Alexa 원본 파일에 증강을 적용합니다.")
    total_files_created = 0

    for wav_path in all_wav_paths:
        file_name = os.path.basename(wav_path)
        base_name, _ = os.path.splitext(file_name)

        try:
            # 오디오 로드 (8kHz)
            audio_data, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
            
            # 증강 적용 및 저장
            count = augment_audio(audio_data, sr, base_name, output_wav_folder)
            total_files_created += count
            
        except Exception as e:
            print(f"파일 처리 오류 발생 ({file_name}): {e}")
            continue

    print("-" * 50)
    print(f"✅ 데이터 증강 완료!")
    print(f"원본 파일 수: {len(all_wav_paths)}개")
    print(f"생성된 총 증강 파일 수: {total_files_created}개")
    print(f"새 파일은 {output_wav_folder}에 저장되었습니다.")