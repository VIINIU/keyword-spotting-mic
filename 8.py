import numpy as np
import os
import glob

# -----------------------------------------------------------------
# [설정] 자네의 스파이크 데이터가 저장된 경로를 지정하게
# -----------------------------------------------------------------
# (이전에 16bit로 생성한 경로를 넣으면 되네)

# # 1. Negative 데이터 경로
spike_path = "C:/Users/11e26/Desktop/internship/source/clear_negative_command/spike_16bit_regenerated"

# 2. Positive 데이터 경로 (Negative를 확인한 뒤, 이 경로로 바꿔서 또 실행해보게)
# spike_path = "C:/Users/11e26/Desktop/internship/source/clear_command_trimmed/spike_16bit_regenerated"
# -----------------------------------------------------------------


def analyze_spike_frequency(path):
    """
    지정된 경로의 모든 .npy 파일(스파이크 트레인)을 로드하여
    평균 스파이크 빈도를 계산합니다.
    """
    print(f"분석 시작: {path}")
    
    # 해당 경로에서 모든 .npy 파일을 찾음
    all_spike_files = glob.glob(os.path.join(path, "*.npy"))
    
    if not all_spike_files:
        print(" [오류] 해당 경로에서 .npy 파일을 찾을 수 없습니다.")
        print(" 'spike_path' 변수가 올바른지 확인하세요.")
        return

    # 각 파일의 평균 빈도를 저장할 리스트
    all_frequencies = []

    for i, file_path in enumerate(all_spike_files):
        # 스파이크 트레인 데이터 로드
        spike_data = np.load(file_path)
        
        if spike_data.size == 0:
            print(f"  - {os.path.basename(file_path)} 파일이 비어있습니다. 건너뜁니다.")
            continue
            
        # 🚨 핵심 로직: 스파이크 빈도 계산
        # 스파이크 데이터는 0(없음) 또는 1(발생)로 구성됨.
        # 따라서 np.mean()을 호출하면 '1의 비율', 즉 평균 발화율(빈도)이 됨.
        file_frequency = np.mean(spike_data)
        
        all_frequencies.append(file_frequency)
        
        # 모든 파일을 다 출력하면 너무 기니까 100개마다 한 번씩만 진행 상황 표시
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}개 파일 처리 완료.")

    # --- 분석 결과 요약 ---
    if not all_frequencies:
        print(" [오류] 유효한 스파이크 파일을 처리하지 못했습니다.")
        return
        
    overall_avg_freq = np.mean(all_frequencies)
    min_freq = np.min(all_frequencies)
    max_freq = np.max(all_frequencies)
    # 0의 비율 (스파이크가 전혀 없는 파일의 비율)
    zero_spike_files = np.sum(np.array(all_frequencies) == 0)
    
    print("\n--- 스파이크 빈도 분석 결과 ---")
    print(f"총 분석 파일 수: {len(all_spike_files)}개")
    print(f"  - 스파이크가 전혀 없는 파일 수: {zero_spike_files}개")
    print(f"최소 빈도 (한 파일 내): {min_freq:.6f}")
    print(f"최대 빈도 (한 파일 내): {max_freq:.6f}")
    print("---------------------------------")
    print(f"✅ 전체 데이터셋의 평균 스파이크 빈도: {overall_avg_freq:.6f}")
    print("---------------------------------")


# --- 메인 실행 ---
if __name__ == "__main__":
    analyze_spike_frequency(spike_path)
    
    # [팁] Positive 데이터 경로도 확인하려면
    # 1. 위 10행의 spike_path를 주석 처리
    # 2. 위 13행의 spike_path 주석을 해제
    # 3. 코드를 다시 실행