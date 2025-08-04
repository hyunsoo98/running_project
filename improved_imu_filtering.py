# ⚠️ bleak 설치 필요: pip install bleak
import asyncio, math, threading, time, tkinter as tk, matplotlib.pyplot as plt, numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bleak import BleakClient
from datetime import datetime
import json, socket
from scipy import signal
from collections import deque

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

# --- 개선된 필터링 클래스들 ---
class ImprovedKalmanFilter:
    """개선된 칼만 필터 - 더 정확한 노이즈 모델링"""
    
    def __init__(self, dt, process_noise=0.01, measurement_noise=1.0):
        self.dt = dt
        
        # 상태 벡터: [angle, angular_velocity, angular_acceleration]
        self.x = np.zeros((3, 1))
        
        # 상태 공분산 행렬
        self.P = np.eye(3) * 1000
        
        # 상태 전이 행렬 (3차 모델)
        self.F = np.array([
            [1, dt, 0.5*dt*dt],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # 측정 행렬
        self.H = np.array([[1, 0, 0]])
        
        # 프로세스 노이즈 공분산 (더 정교한 모델)
        self.Q = np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2, dt],
            [dt**2/2, dt, 1]
        ]) * process_noise
        
        # 측정 노이즈 공분산
        self.R = np.array([[measurement_noise]])
        
    def predict(self):
        """예측 단계"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """업데이트 단계"""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        
    def get_angle(self):
        return float(self.x[0, 0])
    
    def get_angular_velocity(self):
        return float(self.x[1, 0])
    
    def get_angular_acceleration(self):
        return float(self.x[2, 0])

class AdaptiveFilter:
    """적응형 필터 - 노이즈 레벨에 따라 자동 조절"""
    
    def __init__(self, window_size=20, initial_alpha=0.1):
        self.window_size = window_size
        self.alpha = initial_alpha
        self.buffer = deque(maxlen=window_size)
        self.prev_filtered = 0.0
        
    def update(self, measurement):
        """적응형 필터링"""
        self.buffer.append(measurement)
        
        if len(self.buffer) < 3:
            return measurement
        
        # 노이즈 레벨 추정
        noise_level = np.std(self.buffer)
        
        # 노이즈 레벨에 따른 알파값 조절
        if noise_level > 5.0:  # 높은 노이즈
            adaptive_alpha = 0.05
        elif noise_level > 2.0:  # 중간 노이즈
            adaptive_alpha = 0.1
        else:  # 낮은 노이즈
            adaptive_alpha = 0.3
        
        # 필터링 적용
        filtered = adaptive_alpha * measurement + (1 - adaptive_alpha) * self.prev_filtered
        self.prev_filtered = filtered
        
        return filtered

class ComplementaryFilter:
    """상보 필터 - 가속도계와 자이로스코프 융합"""
    
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.prev_angle = 0.0
        
    def update(self, accel_angle, gyro_rate, dt):
        """상보 필터 업데이트"""
        # 자이로스코프 적분
        gyro_angle = self.prev_angle + gyro_rate * dt
        
        # 상보 필터 적용
        filtered_angle = self.alpha * gyro_angle + (1 - self.alpha) * accel_angle
        self.prev_angle = filtered_angle
        
        return filtered_angle

class OutlierDetector:
    """이상치 탐지 및 제거"""
    
    def __init__(self, window_size=10, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = deque(maxlen=window_size)
        self.prev_valid = 0.0
        
    def detect_and_remove(self, measurement):
        """이상치 탐지 및 제거"""
        self.buffer.append(measurement)
        
        if len(self.buffer) < 3:
            return measurement
        
        mean_val = np.mean(self.buffer)
        std_val = np.std(self.buffer)
        
        # 이상치 탐지
        if abs(measurement - mean_val) > self.threshold * std_val:
            # 이상치인 경우 이전 유효값 반환
            return self.prev_valid
        
        self.prev_valid = measurement
        return measurement

# --- 데이터 구조 ---
history_len = 100
imu_data = {name: {k: [0]*history_len for k in ['Roll','Pitch','Yaw']} for name in DEVICES}
data_lock = threading.Lock()
offsets = {name: {'Roll':0,'Pitch':0,'Yaw':0} for name in DEVICES}
calibrated = {name: False for name in DEVICES}
calibration_data = {name: [] for name in DEVICES}
structured_data = {name: [] for name in DEVICES}

# --- 개선된 필터 초기화 ---
dt = 0.05
sample_rate = 1.0 / dt

# 각 장치별 필터 초기화
filters = {}
for device_name in DEVICES:
    filters[device_name] = {
        'kalman': {
            'Roll': ImprovedKalmanFilter(dt, process_noise=0.01, measurement_noise=1.0),
            'Pitch': ImprovedKalmanFilter(dt, process_noise=0.01, measurement_noise=1.0),
            'Yaw': ImprovedKalmanFilter(dt, process_noise=0.01, measurement_noise=1.0)
        },
        'adaptive': {
            'Roll': AdaptiveFilter(window_size=20, initial_alpha=0.1),
            'Pitch': AdaptiveFilter(window_size=20, initial_alpha=0.1),
            'Yaw': AdaptiveFilter(window_size=20, initial_alpha=0.1)
        },
        'complementary': {
            'Roll': ComplementaryFilter(alpha=0.98),
            'Pitch': ComplementaryFilter(alpha=0.98),
            'Yaw': ComplementaryFilter(alpha=0.98)
        },
        'outlier_detector': {
            'Roll': OutlierDetector(window_size=10, threshold=3.0),
            'Pitch': OutlierDetector(window_size=10, threshold=3.0),
            'Yaw': OutlierDetector(window_size=10, threshold=3.0)
        }
    }

def calculate_orientation_improved(device_name, ax, ay, az, gx, gy, gz, mx, my):
    """개선된 방향 계산 함수"""
    # 가속도계 기반 각도 계산
    acc_roll = np.degrees(np.arctan2(ay, az))
    acc_pitch = np.degrees(np.arctan2(-ax, math.hypot(ay, az)))
    mag_yaw = np.degrees(np.arctan2(my, mx))
    
    # 자이로스코프 각속도 (rad/s)
    gyro_roll_rate = np.radians(gx)
    gyro_pitch_rate = np.radians(gy)
    gyro_yaw_rate = np.radians(gz)
    
    # 각 축별 필터링 적용
    filtered_angles = {}
    
    for axis, (acc_angle, gyro_rate) in zip(['Roll', 'Pitch', 'Yaw'], 
                                           [(acc_roll, gyro_roll_rate), 
                                            (acc_pitch, gyro_pitch_rate), 
                                            (mag_yaw, gyro_yaw_rate)]):
        
        # 1. 이상치 탐지 및 제거
        clean_angle = filters[device_name]['outlier_detector'][axis].detect_and_remove(acc_angle)
        
        # 2. 상보 필터 적용
        complementary_angle = filters[device_name]['complementary'][axis].update(clean_angle, gyro_rate, dt)
        
        # 3. 적응형 필터 적용
        adaptive_angle = filters[device_name]['adaptive'][axis].update(complementary_angle)
        
        # 4. 칼만 필터 적용
        kalman_filter = filters[device_name]['kalman'][axis]
        kalman_filter.predict()
        kalman_filter.update(np.array([[adaptive_angle]]))
        filtered_angles[axis] = kalman_filter.get_angle()
    
    return filtered_angles['Roll'], filtered_angles['Pitch'], filtered_angles['Yaw']

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

        # 개선된 방향 계산 사용
        roll, pitch, yaw = calculate_orientation_improved(device_name, *acc, *gyr, *mag)
        
        with data_lock:
            if not calibrated[device_name]:
                calibration_data[device_name].append((roll, pitch, yaw))
                if len(calibration_data[device_name]) >= 50:
                    for i, key in enumerate(['Roll','Pitch','Yaw']):
                        offsets[device_name][key] = sum(d[i] for d in calibration_data[device_name]) / len(calibration_data[device_name])
                    calibrated[device_name] = True
                    print(f"✅ {device_name} 캘리브레이션 완료:", offsets[device_name])
            else:
                # 오프셋 제거
                roll -= offsets[device_name]['Roll']
                pitch -= offsets[device_name]['Pitch']
                yaw -= offsets[device_name]['Yaw']
                
                # 데이터 업데이트
                for key, val in zip(['Roll','Pitch','Yaw'], [roll, pitch, yaw]):
                    imu_data[device_name][key].append(val)
                    if len(imu_data[device_name][key]) > history_len:
                        imu_data[device_name][key].pop(0)
                
                # 구조화된 데이터 저장
                structured_data[device_name].append({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "Roll": round(roll, 2),
                    "Pitch": round(pitch, 2),
                    "Yaw": round(yaw, 2),
                    "filter_type": "improved_kalman_adaptive"
                })
                
                # UDP 전송
                send_udp({
                    "device": device_name,
                    "roll": round(roll, 2),
                    "pitch": round(pitch, 2),
                    "yaw": round(yaw, 2),
                    "filter_type": "improved"
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
root.title("IMU 실시간 시각화 - 개선된 필터링")
fig, axes = plt.subplots(4, 3, figsize=(12, 8))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

lines = {}
for i, device in enumerate(DEVICES):
    for j, key in enumerate(['Roll','Pitch','Yaw']):
        ax = axes[i][j]
        ax.set_title(f"{device} - {key} (개선된 필터)")
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
    filename = f"imu_log_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    print(f"✅ 개선된 필터링 JSON 저장 완료: {filename}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

for name, addr in DEVICES.items():
    start_ble_thread(name, addr)

root.after(100, update_graph)
root.mainloop() 