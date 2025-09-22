import serial
import time

PORT = "COM4"
BAUD = 115200

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"[OK] Opened {PORT} at {BAUD} baud")
    except Exception as e:
        print(f"[ERR] Failed to open {PORT}: {e}")
        return

    try:
        while True:
            # 문자열 전송
            msg = "Hello!\n"
            ser.write(msg.encode("utf-8"))
            print(f"[TX] {msg.strip()}")
            time.sleep(0.1)  
            if ser.in_waiting > 0:
                rx = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
                print(f"[RX] {rx.strip()}")
            else:
                print("[RX] (no data)")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
