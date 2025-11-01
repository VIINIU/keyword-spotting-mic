import serial
import time
import os

# ----------------------------------------------------
# 테스트 환경 설정 (현재 FPGA의 UART 파라미터를 따라야 함)
# ----------------------------------------------------
SERIAL_PORT = 'COM4'    # 실제 COM 포트로 변경하십시오.
BAUD_RATE = 230400      # FPGA에 설정된 Baud Rate (이전 UART 모듈 기준)
TEST_STRING = "Hello FPGA! Testing 1234567890. Communication Check Complete. ZYXWVU." 

# ----------------------------------------------------
def run_text_echo_test_store_forward():
    """
    UART를 통해 텍스트를 보내고, FPGA의 RX 완료 + 2초 Delay 후에 수신을 시작합니다.
    """
    sent_data = TEST_STRING.encode('ascii')
    num_bytes_to_send = len(sent_data)
    received_bytes = bytearray()
    
    # 69 바이트 전송에 걸리는 시간 계산 (오버헤드 포함)
    # (10 bits / 8 bits) * 69 bytes / 115200 bps ≈ 0.006 seconds
    TX_TIME_EST = (num_bytes_to_send * 10) / BAUD_RATE 
    WAIT_TIME_FPGA = 2.0  # Verilog의 T_WAIT_CYCLES_2SEC (2초)
    
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=0.1 
        )
        print(f"\n성공: {SERIAL_PORT} @ {BAUD_RATE}bps 포트 열림.")
        ser.flushInput() 
        ser.flushOutput() 
        time.sleep(1) 

        # --- A. 데이터 전송 (FPGA는 S_RECORD 상태에서 기록) ---
        start_time = time.time()
        print(f"--- 테스트 문자열 전송 시작 ({num_bytes_to_send} 바이트) ---")
        
        # PC TX 버퍼에 한 번에 넣어 전송 시간을 최소화
        ser.write(sent_data) 
        
        end_time = time.time()
        actual_tx_duration = end_time - start_time
        print(f"전송 완료. 소요 시간: {actual_tx_duration:.4f}초.")

        # --- B. FPGA RX 완료 + 2초 Delay 대기 ---
        # Verilog FSM이 S_RECORD -> S_WAIT_2SEC -> S_PLAYBACK으로 넘어가기를 기다립니다.
        TOTAL_WAIT_DURATION = actual_tx_duration + WAIT_TIME_FPGA + 0.1 # 안전 마진 100ms 추가
        
        print(f"FPGA Delay 대기 시작 ({TOTAL_WAIT_DURATION:.3f}초)...")
        time.sleep(TOTAL_WAIT_DURATION) 
        print("Delay 완료. Echo 수신 시작.")

        # --- C. Echo 데이터 수신 ---
        # FPGA는 이제 8kHz 속도로 데이터를 내보내기 시작해야 합니다.
        time.sleep(0.5) # FPGA가 데이터를 모아서 버퍼에 쌓을 시간
        
        # 최대한 모든 데이터를 수신합니다.
        if ser.in_waiting > 0:
            data_chunk = ser.read(ser.in_waiting)
            received_bytes.extend(data_chunk)

        # 텍스트 에코는 매우 빠르므로, 추가적인 수신 루프 실행
        time.sleep(0.5) 
        if ser.in_waiting > 0:
            data_chunk = ser.read(ser.in_waiting)
            received_bytes.extend(data_chunk)


        # --- 결과 검증 ---
        received_string = received_bytes.decode('ascii', errors='ignore')
        
        print(f"\n총 {len(received_bytes)} 바이트 수신.")
        print(f"수신 문자열: {received_string}")
        
        if received_string == TEST_STRING:
            print("\n✅ 테스트 성공: 전송된 문자열과 수신된 에코가 완벽히 일치합니다.")
        else:
            print(f"\n❌ 테스트 실패: 수신된 문자열이 일치하지 않습니다.")
            print(f"   오류 유형: 바이트 수 {num_bytes_to_send} != {len(received_bytes)}")
            print("   이는 FPGA가 TX 재생 상태로 진입하지 못했거나, 전송을 중단했음을 의미합니다.")

    except serial.SerialException as e:
        print(f"시리얼 통신 오류 발생: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print(f"{SERIAL_PORT} 포트 닫힘.")

if __name__ == "__main__":
    run_text_echo_test_store_forward()