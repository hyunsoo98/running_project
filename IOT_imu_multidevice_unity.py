# ⚠️ bleak 설치 필요: pip install bleak
import asyncio, math, threading, time, tkinter as tk, matplotlib.pyplot as plt, numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bleak import BleakClient
from datetime import datetime
import json, socket

# --- BLE 장치 목록 ---
DEVICES = {
    "Rev_1": "64:5B:B3:8B:A5:08",
    "Rev_2": "42:49:B4:0F:5F:B0",
    "Rev_3": "58:BF:25:9B:F2:02",
    "Rev_4": "EC:62:60:82:DB:DA"
}
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1213"

# --- UDP 설정 ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp(data_dict):
    msg = json.dumps(data_dict).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))

# --- 데이터 구조 ---
history_len = 100
imu_data = {name: {k: [0]*history_len for k in ['Roll','Pitch','Yaw']} for name in DEVICES}
data_lock = threading.Lock()
offsets = {name: {'Roll':0,'Pitch':0,'Yaw':0} for name in DEVICES}
calibrated = {name: False for name in DEVICES}
calibration_data = {name: [] for name in DEVICES}
structured_data = {name: [] for name in DEVICES}

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((2,1)); self.P = np.eye(2)
        self.F = np.array([[1,dt],[0,1]]); self.H = np.eye(2)
        self.Q = np.eye(2)*0.01; self.R = np.eye(2)*1.0
    def predict(self):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        z = z.reshape((2,1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
    def get_angle(self):
        return float(self.x[0,0])

dt = 0.05
filters = {name: {'Roll': KalmanFilter(dt), 'Pitch': KalmanFilter(dt), 'Yaw': KalmanFilter(dt)} for name in DEVICES}

def calculate_orientation_kalman(device_name, ax, ay, az, gx, gy, gz, mx, my):
    acc_roll = np.degrees(np.arctan2(ay, az))
    acc_pitch = np.degrees(np.arctan2(-ax, math.hypot(ay, az)))
    mag_yaw = np.degrees(np.arctan2(my, mx))
    kfs = filters[device_name]
    for axis, val in zip(['Roll','Pitch','Yaw'], [(acc_roll, gx), (acc_pitch, gy), (mag_yaw, gz)]):
        kfs[axis].predict(); kfs[axis].update(np.array(val))
    return kfs['Roll'].get_angle(), kfs['Pitch'].get_angle(), kfs['Yaw'].get_angle()

def make_notification_handler(device_name):
    def handler(sender, data):
        s = data.decode(errors='ignore').strip()
        if not (s.startswith("A=") or s.startswith("A:")): return
        acc = gyr = mag = None
        try:
            if s.startswith("A="):
                parts = s.split(";")
                acc = list(map(float, parts[0].split("=")[1].split("|")))
                gyr = list(map(float, parts[1].split("=")[1].split("|")))
                mag = list(map(float, parts[2].split("=")[1].split("|")[:2]))
            elif s.startswith("A:"):
                acc_part, gyr_part = s.split("G:")
                acc = list(map(float, acc_part.replace("A:", "").split(",")))
                gyr = list(map(float, gyr_part.strip().split(",")))
                mag = [0.0, 0.0]
        except: return

        roll, pitch, yaw = calculate_orientation_kalman(device_name, *acc, *gyr, *mag)
        with data_lock:
            if not calibrated[device_name]:
                calibration_data[device_name].append((roll, pitch, yaw))
                if len(calibration_data[device_name]) >= 50:
                    for i, key in enumerate(['Roll','Pitch','Yaw']):
                        offsets[device_name][key] = sum(d[i] for d in calibration_data[device_name]) / len(calibration_data[device_name])
                    calibrated[device_name] = True
                    print(f"✅ {device_name} 캘리브레이션 완료:", offsets[device_name])
            else:
                roll -= offsets[device_name]['Roll']
                pitch -= offsets[device_name]['Pitch']
                yaw -= offsets[device_name]['Yaw']
                for key, val in zip(['Roll','Pitch','Yaw'], [roll, pitch, yaw]):
                    imu_data[device_name][key].append(val)
                    if len(imu_data[device_name][key]) > history_len:
                        imu_data[device_name][key].pop(0)
                structured_data[device_name].append({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "Roll": round(roll,2),
                    "Pitch": round(pitch,2),
                    "Yaw": round(yaw,2)
                })
                send_udp({
                    "device": device_name,
                    "roll": round(roll, 2),
                    "pitch": round(pitch, 2),
                    "yaw": round(yaw, 2)
                })
    return handler

async def ble_loop(device_name, address):
    print(f"🔍 {device_name}({address}) 연결 시도 중...")
    try:
        async with BleakClient(address) as client:
            if await client.is_connected():
                print(f"✅ {device_name} 연결 성공")
                await client.start_notify(CHARACTERISTIC_UUID, make_notification_handler(device_name))
                while True:
                    await asyncio.sleep(dt)
    except Exception as e:
        print(f"❌ {device_name} 예외 발생: {e}")

def start_ble_thread(device_name, address):
    def retry_loop():
        while True:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(ble_loop(device_name, address))
            except Exception as e:
                print(f"⚠️ {device_name} 스레드 오류: {e} — 5초 후 재시도")
                time.sleep(5)
    threading.Thread(target=retry_loop, daemon=True).start()

# --- GUI ---
root = tk.Tk()
root.geometry("1200x800")
root.title("IMU 실시간 시각화 - 4기기")
fig, axes = plt.subplots(4, 3, figsize=(12, 8))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

lines = {}
for i, device in enumerate(DEVICES):
    for j, key in enumerate(['Roll','Pitch','Yaw']):
        ax = axes[i][j]
        ax.set_title(f"{device} - {key}")
        ax.set_ylim(-180, 180)
        ax.set_xlim(0, history_len)
        lines[(device, key)], = ax.plot(range(history_len), imu_data[device][key])

def update_graph():
    with data_lock:
        for device in DEVICES:
            for key in ['Roll','Pitch','Yaw']:
                lines[(device, key)].set_ydata(imu_data[device][key])
    canvas.draw_idle()
    root.after(100, update_graph)

def on_close():
    filename = f"imu_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    print(f"✅ JSON 저장 완료: {filename}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

for name, addr in DEVICES.items():
    start_ble_thread(name, addr)

root.after(100, update_graph)
root.mainloop()
