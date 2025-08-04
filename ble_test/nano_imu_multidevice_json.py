import asyncio
import math
import threading
import time
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bleak import BleakClient
from datetime import datetime
import json
from queue import Queue, Empty

# --- BLE 장치 목록 ---
DEVICES = {
    "Rev_1": "64:5B:B3:8B:A5:08",
    "Rev_2": "42:49:B4:0F:5F:B0"
}
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1213"

# --- 데이터 저장 공간 ---
history_len = 100
imu_data = {
    name: {k: [0]*history_len for k in ['Roll','Pitch','Yaw','Speed']}
    for name in DEVICES
}
data_lock = threading.Lock()
structured_data = {name: [] for name in DEVICES}
imu_queue = {name: Queue() for name in DEVICES}

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

dt = 0.033
filters = {name: {
    'Roll': KalmanFilter(dt),
    'Pitch': KalmanFilter(dt),
    'Yaw': KalmanFilter(dt)
} for name in DEVICES}

def calculate_orientation_kalman(device_name, ax, ay, az, gx, gy, gz, mx=0.0, my=0.0):
    acc_roll = np.degrees(np.arctan2(ay, az))
    acc_pitch = np.degrees(np.arctan2(-ax, math.hypot(ay, az)))
    mag_yaw = 0.0
    kfs = filters[device_name]
    for axis, val in zip(['Roll','Pitch','Yaw'], [(acc_roll, gx), (acc_pitch, gy), (mag_yaw, gz)]):
        kfs[axis].predict(); kfs[axis].update(np.array(val))
    return kfs['Roll'].get_angle(), kfs['Pitch'].get_angle(), kfs['Yaw'].get_angle()

def make_notification_handler(device_name):
    def handler(sender, data):
        imu_queue[device_name].put(data.decode(errors='ignore'))
    return handler

def process_imu_data(device_name):
    while True:
        try:
            s = imu_queue[device_name].get_nowait()
        except Empty:
            time.sleep(0.005)
            continue

        if not (s.startswith("A=") or s.startswith("A:")):
            continue

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
                mag = [0.0, 0.0, 0.0]
        except: continue

        roll_v, pitch_v, yaw_v = calculate_orientation_kalman(device_name, *acc, *gyr, *mag)
        speed = (math.sqrt(sum(a*a for a in acc)) - 1) * 9.8 * dt

        with data_lock:
            for key, val in zip(['Roll','Pitch','Yaw','Speed'], [roll_v, pitch_v, yaw_v, speed]):
                imu_data[device_name][key].append(val)
                if len(imu_data[device_name][key]) > history_len:
                    imu_data[device_name][key].pop(0)
            structured_data[device_name].append({
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "Roll": round(roll_v,2),
                "Pitch": round(pitch_v,2),
                "Yaw": round(yaw_v,2),
                "Speed": round(speed,4)
            })
            structured_data[device_name] = structured_data[device_name][-300:]

async def ble_loop(device_name, address):
    while True:
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
            print(f"⚠️ {device_name} 재시도 중: {e}")
            await asyncio.sleep(2)

def start_ble_thread(device_name, address):
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ble_loop(device_name, address))
    threading.Thread(target=run_loop, daemon=True).start()
    threading.Thread(target=process_imu_data, args=(device_name,), daemon=True).start()

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

def update_graph():
    with data_lock:
        for device in DEVICES:
            for key in ['Roll','Pitch','Yaw','Speed']:
                lines[(device, key)].set_ydata(imu_data[device][key])
    canvas.draw_idle()
    root.after(33, update_graph)

def on_close():
    filename = f"imu_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    print(f"✅ JSON 저장 완료: {filename}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

for name, addr in DEVICES.items():
    start_ble_thread(name, addr)

root.after(33, update_graph)
root.mainloop()
