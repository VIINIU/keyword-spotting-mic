import librosa
import soundfile as sf
import os
import glob
import numpy as np

# ====================================================
# A. 설정 변수
# ====================================================
SAMPLE_RATE = 8000 # 학습 시 사용한 샘플링 레이트와 동일해야 함
TOP_DB = 30 # 🚨 침묵을 정의하는 민감도 (데시벨). 조정 필요!
                         # 숫자가 낮을수록(e.g., 20) 작은 소리도 침묵으로 간주하여 더 많이 자름.
                         # 숫자가 높을수록(e.g., 60) 큰 침묵만 자름.

# 원본 WAV 파일이 있는 폴더 경로 (Alexa 또는 Non-Alexa 폴더 경로로 변경)
SOURCE_WAV_FOLDER = "C:/Users/11e26/Desktop/internship/source/clear_negative_command" 
# 트림된 파일을 저장할 폴더 경로
OUTPUT_WAV_FOLDER = "C:/Users/11e26/Desktop/internship/source/clear_negative_command_trimmed" 

# ----------------------------------------------------
# B. 침묵 트림 및 저장 함수
# ----------------------------------------------------
def trim_and_save(input_path, output_dir, top_db, sr):
    """주어진 WAV 파일의 앞뒤 침묵을 제거하고 새로운 파일로 저장합니다."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 오디오 로드
    try:
        y, _ = librosa.load(input_path, sr=sr)
    except Exception as e:
        print(f"오류: {input_path} 로드 실패 - {e}")
        return False, 0, 0

    # 2. 침묵 트림 실행
    # librosa.effects.trim은 오디오 데이터와 (시작, 끝) 인덱스를 반환합니다.
    y_trimmed, index = librosa.effects.trim(y, top_db=top_db)

    # 3. 파일 저장
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    sf.write(output_path, y_trimmed, sr)
    
    # 4. 길이 계산
    original_duration = librosa.get_duration(y=y, sr=sr)
    trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)
    
    return True, original_duration, trimmed_duration

# ----------------------------------------------------
# C. 메인 실행 블록
# ----------------------------------------------------
if __name__=="__main__":
    
    all_wav_paths = glob.glob(os.path.join(SOURCE_WAV_FOLDER, "*.wav"))
    
    if not all_wav_paths:
        print(f"경고: {SOURCE_WAV_FOLDER} 경로에서 WAV 파일을 찾을 수 없습니다.")
        exit()

    print(f"총 {len(all_wav_paths)}개의 WAV 파일에서 침묵을 트림합니다 (TOP_DB={TOP_DB}).")
    print("-" * 50)
    
    success_count = 0
    total_time_saved = 0.0

    for i, wav_path in enumerate(all_wav_paths):
        file_name = os.path.basename(wav_path)
        
        is_success, orig_dur, trim_dur = trim_and_save(wav_path, OUTPUT_WAV_FOLDER, TOP_DB, SAMPLE_RATE)
        
        if is_success:
            time_saved = orig_dur - trim_dur
            total_time_saved += time_saved
            success_count += 1
            
            if (i % 50 == 0) or (i == len(all_wav_paths) - 1):
                print(f"[{i+1}/{len(all_wav_paths)}] {file_name}: 원본 {orig_dur:.2f}s -> 트림 {trim_dur:.2f}s (절약: {time_saved:.2f}s)")
        
    print("-" * 50)
    print(f"✅ 전체 트림 완료. 성공 파일 수: {success_count}개")
    print(f"총 절약된 시간: {total_time_saved:.2f}초")