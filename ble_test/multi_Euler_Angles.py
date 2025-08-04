
import asyncio
import math
import threading
import time
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bleak import BleakClient

# --- BLE 장치 목록 ---
DEVICES = {
    "Rev_2": "42:49:B4:0F:5F:B0",
    "Rev_3": "58:BF:25:9B:F2:02",
    "Rev_4": "EC:62:60:82:DB:DA"
}
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1213"

# --- 데이터 저장 공간 ---
history_len = 100
imu_data = {
    name: {k: [0]*history_len for k in ['Roll','Pitch','Yaw','Speed']}
    for name in DEVICES
}
data_lock = threading.Lock()
offsets = {name: {'Roll':0,'Pitch':0,'Yaw':0} for name in DEVICES}
calibrated = {name: False for name in DEVICES}
calibration_data = {name: [] for name in DEVICES}

# Kalman 필터 클래스
class KalmanFilter:
    def __init__(self, dt, process_noise=0.01, measurement_noise=1.0):
        self.dt = dt
        self.x = np.zeros((2,1)); self.P = np.eye(2)
        self.F = np.array([[1,dt],[0,1]])
        self.H = np.eye(2)
        self.Q = np.eye(2)*process_noise
        self.R = np.eye(2)*measurement_noise
    def predict(self):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        z = z.reshape((2,1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
    def get_angle(self):
        return float(self.x[0,0])

dt = 0.05
filters = {name: {
    'Roll': KalmanFilter(dt),
    'Pitch': KalmanFilter(dt),
    'Yaw': KalmanFilter(dt)
} for name in DEVICES}

def calculate_orientation_kalman(device_name, ax, ay, az, gx, gy, gz, mx, my):
    acc_roll = np.degrees(np.arctan2(ay, az))
    acc_pitch = np.degrees(np.arctan2(-ax, math.hypot(ay, az)))
    mag_yaw = np.degrees(np.arctan2(my, mx))
    kfs = filters[device_name]
    for axis, val in zip(['Roll','Pitch','Yaw'], [(acc_roll, gx), (acc_pitch, gy), (mag_yaw, gz)]):
        kfs[axis].predict(); kfs[axis].update(np.array(val))
    return kfs['Roll'].get_angle(), kfs['Pitch'].get_angle(), kfs['Yaw'].get_angle()

# --- BLE Notify 핸들러 생성기 ---
def make_notification_handler(device_name):
    def handler(sender, data):
        s = data.decode(errors='ignore').strip()
        print(f"[{device_name}] 수신됨 →", s)

        try:
            if s.startswith("A=") and ";" in s:  # Rev_2 포맷
                parts = s.split(";")
                if len(parts) < 3: return
                acc = list(map(float, parts[0].split("=")[1].split("|")))
                gyr = list(map(float, parts[1].split("=")[1].split("|")))
                mag = list(map(float, parts[2].split("=")[1].split("|")[:2]))  # mz는 무시
            elif "A:" in s and "G:" in s:  # Rev_3, Rev_4 포맷
                acc_str, gyr_str = s.split("G:")
                acc = list(map(float, acc_str.replace("A:", "").strip().split(",")))
                gyr = list(map(float, gyr_str.strip().split(",")))
                mag = [0.0, 0.0]
            else:
                return
        except Exception as e:
            print(f"[{device_name}] ❌ 포맷 오류:", e)
            return

        roll_v, pitch_v, yaw_v = calculate_orientation_kalman(device_name, *acc, *gyr, *mag)

        with data_lock:
            if not calibrated[device_name]:
                calibration_data[device_name].append((roll_v, pitch_v, yaw_v))
                if len(calibration_data[device_name]) >= 50:
                    for i, key in enumerate(['Roll','Pitch','Yaw']):
                        offsets[device_name][key] = sum(d[i] for d in calibration_data[device_name]) / 50
                    calibrated[device_name] = True
                    print(f"✅ {device_name} Calibration done:", offsets[device_name])
            else:
                roll_v -= offsets[device_name]['Roll']
                pitch_v -= offsets[device_name]['Pitch']
                yaw_v -= offsets[device_name]['Yaw']
                imu_data[device_name]['Roll'].append(roll_v)
                imu_data[device_name]['Pitch'].append(pitch_v)
                imu_data[device_name]['Yaw'].append(yaw_v)
                acc_norm = math.sqrt(sum(a*a for a in acc))
                imu_data[device_name]['Speed'].append((acc_norm - 1) * 9.8 * dt)
                for key in imu_data[device_name]:
                    if len(imu_data[device_name][key]) > history_len:
                        imu_data[device_name][key].pop(0)
    return handler

# --- BLE 수신 루프 ---
async def ble_loop(device_name, address):
    print(f"🔍 {device_name}({address}) 연결 시도 중...")
    try:
        async with BleakClient(address) as client:
            connected = await client.is_connected()
            if connected:
                print(f"✅ {device_name} 연결 성공")
            else:
                print(f"❌ {device_name} 연결 실패")
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

# --- GUI 설정 ---
root = tk.Tk()
root.geometry("1600x800")
root.title("IMU Real-time Plot - Multi Device")

fig, axes = plt.subplots(3, 4, figsize=(16, 8))
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
    ax_speed = axes[i][3]
    ax_speed.set_title(f"{device} - Speed")
    ax_speed.set_ylim(-5, 5)
    lines[(device, 'Speed')], = ax_speed.plot(range(history_len), imu_data[device]['Speed'])

# --- 주기적 업데이트 ---
def update_graph():
    with data_lock:
        for device in DEVICES:
            for key in ['Roll','Pitch','Yaw','Speed']:
                lines[(device, key)].set_ydata(imu_data[device][key])
    canvas.draw_idle()
    root.after(100, update_graph)

# --- BLE 스레드 시작 ---
for name, addr in DEVICES.items():
    start_ble_thread(name, addr)

root.after(100, update_graph)
root.mainloop()
