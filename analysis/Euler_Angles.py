import asyncio
import math
import threading
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bleak import BleakClient

# --- BLE 설정 ---
ADDRESS = "42:49:B4:0F:5F:B0"
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"

# --- 데이터 저장 공간 ---
history_len = 100
imu_data = {k: [0]*history_len for k in ['Roll','Pitch','Yaw','Speed']}
data_lock = threading.Lock()

# 캘리브레이션
offsets = {'Roll':0,'Pitch':0,'Yaw':0}
calibrated = False
calibration_data = []

# Kalman 필터 클래스 (변경 없음)
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
kf_roll = KalmanFilter(dt); kf_pitch = KalmanFilter(dt); kf_yaw = KalmanFilter(dt)

def calculate_orientation_kalman(ax, ay, az, gx, gy, gz, mx, my):
    acc_roll = np.degrees(np.arctan2(ay, az))
    acc_pitch = np.degrees(np.arctan2(-ax, math.hypot(ay, az)))
    mag_yaw = np.degrees(np.arctan2(my, mx))
    for kf, meas in zip((kf_roll, kf_pitch, kf_yaw),
                        ((acc_roll, gx), (acc_pitch, gy), (mag_yaw, gz))):
        kf.predict(); kf.update(np.array(meas))
    return kf_roll.get_angle(), kf_pitch.get_angle(), kf_yaw.get_angle()

# --- BLE Notify 핸들러: 데이터 저장만 수행 ---
def handle_notification(sender, data):
    global calibrated
    s = data.decode(errors='ignore').strip()
    if not s.startswith("A="): return

    parts = s.split(";")
    if len(parts) != 3: return

    def parse(token, cnt):
        arr = list(map(float, token.split("|")))
        if len(arr) != cnt: raise ValueError
        return arr

    try:
        acc = parse(parts[0].split("=")[1], 3)
        gyr = parse(parts[1].split("=")[1], 3)
        mag = parse(parts[2].split("=")[1], 2)
    except:
        return

    roll_v, pitch_v, yaw_v = calculate_orientation_kalman(*acc, *gyr, *mag)

    with data_lock:
        if not calibrated:
            calibration_data.append((roll_v, pitch_v, yaw_v))
            if len(calibration_data) >= 50:
                for i, key in enumerate(['Roll','Pitch','Yaw']):
                    offsets[key] = sum(d[i] for d in calibration_data) / len(calibration_data)
                calibrated = True
                print("✅ Calibration done:", offsets)
        else:
            # 오프셋 적용
            roll_v -= offsets['Roll']
            pitch_v -= offsets['Pitch']
            yaw_v -= offsets['Yaw']
            imu_data['Roll'].append(roll_v)
            imu_data['Pitch'].append(pitch_v)
            imu_data['Yaw'].append(yaw_v)
            acc_norm = math.sqrt(sum(a*a for a in acc))
            imu_data['Speed'].append((acc_norm - 1) * 9.8 * dt)

        for key in imu_data:
            if len(imu_data[key]) > history_len:
                imu_data[key].pop(0)

# --- BLE 수신 루프 ---
async def ble_loop():
    async with BleakClient(ADDRESS) as client:
        await client.start_notify(CHARACTERISTIC_UUID, handle_notification)
        while True:
            await asyncio.sleep(dt)

def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_loop())

# --- GUI 설정 ---
root = tk.Tk()
root.geometry("1200x600")
root.title("IMU Real‑time Plot")

fig = plt.Figure(figsize=(12,5))
gs = fig.add_gridspec(2,5)
axes = [fig.add_subplot(gs[0,i]) for i in range(3)]
axes += [fig.add_subplot(gs[:,3]), fig.add_subplot(gs[:,4])]
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

lines = {}
for i, key in enumerate(['Roll','Pitch','Yaw']):
    ax = axes[i]
    ax.set_title(key); ax.set_xlim(0,history_len); ax.set_ylim(-180,180)
    ax.set_xlabel('Time'); ax.set_ylabel('°')
    lines[key], = ax.plot(range(history_len), imu_data[key], lw=1)

axes[3].set_title('Speed'); axes[4].set_title('Calib Progress')
bar = axes[4].bar(['Progress'], [0], color='orange')
axes[4].set_ylim(0,100)

# --- 주기적 업데이트 함수 ---
def update_graph():
    with data_lock:
        for key in ['Roll','Pitch','Yaw']:
            lines[key].set_ydata(imu_data[key])
        axes[3].cla()
        axes[3].bar(['Speed'], [imu_data['Speed'][-1]], color='skyblue')
        axes[3].set_ylim(-5,5); axes[3].set_title('Speed')
        if not calibrated:
            bar[0].set_height(len(calibration_data)/50*100)
    canvas.draw_idle()
    root.after(100, update_graph)  # 0.1초마다 갱신

# --- 실행 ---
threading.Thread(target=start_ble_thread, daemon=True).start()
root.after(100, update_graph)
root.mainloop()
