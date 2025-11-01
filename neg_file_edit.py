import librosa
import soundfile as sf
import os
import numpy as np
import random
import glob # 파일 목록 검색을 위해 glob 라이브러리 추가

# ----------------------------------------------------
# 1. 설정 변수
# ----------------------------------------------------
# 원본 Negative MP3/WAV 파일이 있는 폴더 경로 (수정 필요)
INPUT_WAV_DIR = "C:/Users/11e26/Desktop/internship/source/non-alexa" 
# 전처리된 파일을 저장할 폴더 경로 (수정 필요)
OUTPUT_WAV_DIR = "C:/Users/11e26/Desktop/internship/source/non-alexa_command" 

TARGET_SR = 8000  # 목표 샘플링 레이트 (8kHz)
DURATION = 2.0     # 목표 길이 (2.0초)
TARGET_SAMPLES = int(TARGET_SR * DURATION) # 16000 샘플
NUM_SAMPLES_TO_PROCESS = 1000 # 20601개 중 처리할 파일 개수 (조정 가능)

# ----------------------------------------------------
# 2. 전처리 함수 (MP3/WAV 모두 처리 가능하도록 수정)
# ----------------------------------------------------
def preprocess_negative_audio():
    # 1. 입력 폴더 확인 및 파일 목록 가져오기
    if not os.path.exists(INPUT_WAV_DIR):
        print(f"오류: 입력 폴더를 찾을 수 없습니다: {INPUT_WAV_DIR}")
        return

    # WAV와 MP3 파일명 목록을 모두 가져옵니다.
    mp3_files = glob.glob(os.path.join(INPUT_WAV_DIR, '*.mp3'))
    
    all_files = mp3_files
    print(f"총 {len(all_files)}개의 WAV/MP3 오디오 파일을 찾았습니다.")

    if len(all_files) == 0:
        print("경고: 입력 폴더에 처리 가능한 WAV 또는 MP3 파일이 없습니다.")
        return

    # 2. 랜덤 추출
    # 전체 파일 경로 중 NUM_SAMPLES_TO_PROCESS 만큼 무작위로 추출
    if len(all_files) > NUM_SAMPLES_TO_PROCESS:
        selected_paths = random.sample(all_files, NUM_SAMPLES_TO_PROCESS)
    else:
        selected_paths = all_files
    
    print(f"랜덤으로 {len(selected_paths)}개의 파일을 선택했습니다.")

    # 3. 출력 폴더 생성
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)

    # 4. 파일 처리 루프
    processed_count = 0
    for input_path in selected_paths:
        
        # 원본 파일명 추출 (확장자 포함)
        original_filename_with_ext = os.path.basename(input_path)
        original_filename_base, _ = os.path.splitext(original_filename_with_ext)
        
        # 출력 파일 이름은 WAV로 통일하고 'neg_' 접두사 추가
        output_filename = f"neg_{original_filename_base}.wav"
        output_path = os.path.join(OUTPUT_WAV_DIR, output_filename)

        try:
            # 4-1. 파일 로드 및 리샘플링
            # librosa.load는 MP3도 지원 (FFmpeg 필요). sr=None으로 로드.
            audio_data, sr = librosa.load(input_path, sr=None)
            
            # 8kHz로 리샘플링
            audio_data_8k = librosa.resample(y=audio_data, orig_sr=sr, target_sr=TARGET_SR)
            
            # 4-2. 트리밍 및 패딩
            
            if audio_data_8k.shape[0] < TARGET_SAMPLES:
                # 길이가 짧으면 0으로 패딩 (균등하게)
                padding_len = TARGET_SAMPLES - audio_data_8k.shape[0]
                pad_left = padding_len // 2
                pad_right = padding_len - pad_left
                trimmed_data = np.pad(audio_data_8k, (pad_left, pad_right), 'constant')
            
            elif audio_data_8k.shape[0] > TARGET_SAMPLES:
                # 길이가 길면 랜덤하게 트리밍
                max_start = audio_data_8k.shape[0] - TARGET_SAMPLES
                start_idx = random.randint(0, max_start) # 랜덤 시작 지점 추출
                trimmed_data = audio_data_8k[start_idx : start_idx + TARGET_SAMPLES]
            
            else:
                trimmed_data = audio_data_8k

            # 4-3. 파일 저장 (출력은 모두 WAV 파일로 통일)
            sf.write(output_path, trimmed_data, TARGET_SR, format='WAV')
            processed_count += 1
            
            if (processed_count % 100 == 0):
                 print(f"진행 상황: {processed_count}/{len(selected_paths)}개 파일 처리 완료")

        except Exception as e:
            print(f"파일 처리 오류 발생 ({original_filename_with_ext}): {e}")
            continue

    print(f"\n--- 전처리 완료 ---")
    print(f"총 {processed_count}개의 전처리된 Negative WAV 파일이 {OUTPUT_WAV_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_negative_audio()