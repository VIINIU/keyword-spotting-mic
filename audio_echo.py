import serial
import time
import numpy as np
import librosa
import soundfile as sf
import os
import struct

# ----------------------------------------------------
# 환경 설정 변수 (115200 Baud로 환원)
# ----------------------------------------------------
SERIAL_PORT = 'COM4'    
BAUD_RATE = 230400      # 8bit 오디오 스트림에 충분
TARGET_SAMPLE_RATE = 8000 # 8kHz
INPUT_FILENAME = "C:/vini_dir/kws_mic/alexa_sample2.wav" 
edited_FILENAME = "C:/vini_dir/kws_mic/alexa_sample_edited.wav" 
OUTPUT_FILENAME = "C:/vini_dir/kws_mic/backed_8bit.wav"  
CHUNK_SIZE = 1024       
TARGET_SAMPLES = 29280 # 예상 샘플 수 (Verilog와 동기화)

# ----------------------------------------------------
# 1. WAV 파일 로드 및 16bit -> 8bit Unsigned Int 변환 (수정)
# ----------------------------------------------------
def load_and_prepare_audio_8bit(file_path):
    """
    16bit 오디오를 로드하여 8bit Unsigned Int (0~255) 바이트로 변환합니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: 입력 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None, None

    print(f"오디오 파일 로드 중: {file_path}")
    audio_data_float, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    
    # ----------------------------------------------------
    # 핵심: 16bit -> 8bit Unsigned Integer (uint8) 변환
    # ----------------------------------------------------
    # 1. float [-1.0, 1.0] -> [0, 1.0] 범위로 이동
    audio_data_scaled = (audio_data_float + 1.0) / 2.0 
    
    # 2. [0, 255] 범위로 스케일링 후 uint8로 변환
    audio_data_int8 = (audio_data_scaled * 255).astype(np.uint8)# ----------------------------------------------------
    # 🚨 8bit 변환 후 WAV 파일로 저장하는 로직 (요청 사항)
    # ----------------------------------------------------
    # 저장 전에 uint8 데이터를 다시 float [-1.0, 1.0]으로 역변환해야 합니다.
    saved_data_float = (audio_data_int8.astype(np.float32) / 255.0 * 2.0) - 1.0
    sf.write(edited_FILENAME, saved_data_float, sr)
    print(f"경고: 8bit로 변환된 오디오가 '{edited_FILENAME}'에 저장되었습니다. 음질을 확인하십시오.")
    
    # 전송할 바이트는 uint8의 raw 바이트 스트림입니다.
    serial_data = audio_data_int8.tobytes()
    num_bytes_to_send = len(audio_data_int8)
    
    print(f"변환 후 샘플 수: {len(audio_data_int8)}")
    print(f"데이터 준비 완료. 전송할 바이트 수 (샘플수 x 1): {num_bytes_to_send}")
    
    return serial_data, num_bytes_to_send

# ----------------------------------------------------
# 2. UART 송신 및 수신 (Store-and-Forward용 수정)
# ----------------------------------------------------
def send_and_receive_echo(serial_data, num_bytes_sent):
    received_bytes = bytearray()
    
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=0.1 
        )
        print(f"\n성공: {SERIAL_PORT} @ {BAUD_RATE}bps 포트 열림.")
        time.sleep(2) 
        ser.flushInput() 
        ser.flushOutput() 

        # --- A. 데이터 전송 ---
        start_time = time.time()
        print("--- 오디오 데이터 전송 시작 ---")
        
        # PC가 전송하는 동안 FPGA는 수신만 합니다. (Store)
        for i in range(0, len(serial_data), CHUNK_SIZE):
            chunk = serial_data[i:i + CHUNK_SIZE]
            ser.write(chunk)
            # PC 버퍼가 넘치지 않도록 짧은 딜레이 사용
            time.sleep(0.005) 
            
        end_time = time.time()
        print(f"전송 완료. {len(serial_data)} 바이트 전송 완료. 소요 시간: {end_time - start_time:.2f}초.")

        # # --- B. FPGA Delay (2초) 대기 및 수신 시작 ---
        # print(f"FPGA 2초 Delay 대기 시작...")
        # # FPGA RX 완료 시간 + 2초 Delay
        # time_to_wait = (end_time - start_time) + 2.5 
        # time.sleep(time_to_wait) 
        # print("Delay 완료. Echo 데이터 수신 시작.")


        # --- C. Echo 데이터 수신 ---
        # FPGA가 Echo를 시작했으므로, 이제 PC는 모두 읽습니다.
        total_received = 0
        read_timeout = 15 
        read_start_time = time.time()

        while total_received < num_bytes_sent and (time.time() - read_start_time) < read_timeout:
            bytes_to_read = ser.in_waiting
            if bytes_to_read > 0:
                data_chunk = ser.read(bytes_to_read)
                received_bytes.extend(data_chunk)
                total_received += len(data_chunk)
                print(f"수신 중... {total_received}/{num_bytes_sent} 바이트 ({total_received/num_bytes_sent*100:.1f}%)", end='\r')
            
            if ser.in_waiting == 0:
                time.sleep(0.001)

        print(f"\n수신 완료. 총 {total_received} 바이트 수신.")

    except serial.SerialException as e:
        print(f"시리얼 통신 오류 발생: {e}")
        return None
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print(f"{SERIAL_PORT} 포트 닫힘.")
            
    return received_bytes

# ----------------------------------------------------
# 3. 수신된 바이트를 WAV 파일로 저장 (8bit용으로 수정)
# ----------------------------------------------------
def save_audio_from_bytes_8bit(byte_data, file_path, sr):
    """
    수신된 8bit unsigned int 바이트 데이터를 float으로 역변환하여 WAV 파일로 저장합니다.
    """
    if not byte_data:
        print("저장할 수신 데이터가 없습니다.")
        return

    received_array_uint8 = np.frombuffer(byte_data, dtype=np.uint8)
    
    # 1. [0, 255] -> [0.0, 1.0] 범위로 스케일링
    received_array_float = received_array_uint8.astype(np.float32) / 255.0
    
    # 2. [0.0, 1.0] -> [-1.0, 1.0] 범위로 역변환
    received_array_final = (received_array_float * 2.0) - 1.0
    
    sf.write(file_path, received_array_final, sr)
    print(f"Echo된 오디오 데이터가 '{file_path}' (샘플 수: {len(received_array_final)})에 저장되었습니다.")

# ----------------------------------------------------
# 메인 실행 블록
# ----------------------------------------------------
if __name__ == "__main__":
    audio_bytes, num_bytes_sent = load_and_prepare_audio_8bit(INPUT_FILENAME)

    if audio_bytes is None:
        exit()

    echoed_bytes = send_and_receive_echo(audio_bytes, num_bytes_sent)
    
    save_audio_from_bytes_8bit(echoed_bytes, OUTPUT_FILENAME, TARGET_SAMPLE_RATE)