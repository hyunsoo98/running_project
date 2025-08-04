# 러닝 모션 최적화 필터링 시스템
import asyncio, math, threading, time, tkinter as tk, matplotlib.pyplot as plt, numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bleak import BleakClient
from datetime import datetime
import json, socket
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

# --- 러닝 모션 최적화 필터들 ---
class AdaptiveKalmanFilter:
    """적응형 칼만 필터 - 러닝 모션에 최적화"""
    
    def __init__(self, dt, base_process_noise=0.1, base_measurement_noise=0.5):
        self.dt = dt
        self.base_process_noise = base_process_noise
        self.base_measurement_noise = base_measurement_noise
        
        # 상태 벡터: [angle, angular_velocity]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 10  # 초기 불확실성 증가
        
        # 상태 전이 행렬
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        
        # 적응형 노이즈 공분산
        self.Q = np.eye(2) * base_process_noise
        self.R = np.array([[base_measurement_noise]])
        
        # 모션 감지용 버퍼
        self.motion_buffer = deque(maxlen=10)
        self.prev_angle = 0.0
        
    def detect_motion_intensity(self, measurement):
        """모션 강도 감지"""
        self.motion_buffer.append(measurement)
        
        if len(self.motion_buffer) < 3:
            return 1.0
        
        # 각도 변화율 계산
        angle_change = abs(measurement - self.prev_angle)
        velocity = np.mean(np.abs(np.diff(list(self.motion_buffer))))
        
        # 모션 강도 계산 (0.1 ~ 3.0)
        motion_intensity = min(3.0, max(0.1, velocity * 10 + angle_change * 2))
        
        self.prev_angle = measurement
        return motion_intensity
    
    def adapt_noise_parameters(self, measurement):
        """모션에 따른 노이즈 파라미터 적응"""
        motion_intensity = self.detect_motion_intensity(measurement)
        
        # 모션 강도에 따른 노이즈 조절
        if motion_intensity > 2.0:  # 빠른 움직임
            process_noise = self.base_process_noise * 3.0
            measurement_noise = self.base_measurement_noise * 0.3
        elif motion_intensity > 1.0:  # 중간 움직임
            process_noise = self.base_process_noise * 1.5
            measurement_noise = self.base_measurement_noise * 0.7
        else:  # 느린 움직임
            process_noise = self.base_process_noise * 0.5
            measurement_noise = self.base_measurement_noise * 1.2
        
        # 노이즈 공분산 업데이트
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])
    
    def predict(self):
        """예측 단계"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """업데이트 단계"""
        # 모션에 따른 노이즈 적응
        self.adapt_noise_parameters(measurement)
        
        # 칼만 필터 업데이트
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
    
    def get_angle(self):
        return float(self.x[0, 0])
    
    def get_velocity(self):
        return float(self.x[1, 0])

class MotionPredictor:
    """모션 예측기 - 러닝 패턴 학습"""
    
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.angle_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        self.prediction_weight = 0.3
        
    def update(self, angle, velocity):
        """모션 패턴 업데이트"""
        self.angle_buffer.append(angle)
        self.velocity_buffer.append(velocity)
    
    def predict_next_angle(self):
        """다음 각도 예측"""
        if len(self.angle_buffer) < 5:
            return None
        
        # 선형 예측
        if len(self.angle_buffer) >= 3:
            recent_angles = list(self.angle_buffer)[-3:]
            recent_velocities = list(self.velocity_buffer)[-3:]
            
            # 속도 기반 예측
            avg_velocity = np.mean(recent_velocities)
            predicted_angle = recent_angles[-1] + avg_velocity * 0.05  # dt=0.05
            
            return predicted_angle
        
        return None

class NaturalMotionFilter:
    """자연스러운 모션 필터"""
    
    def __init__(self, dt):
        self.dt = dt
        self.kalman = AdaptiveKalmanFilter(dt)
        self.predictor = MotionPredictor()
        self.smoothing_factor = 0.7
        
        # 자연스러운 움직임을 위한 파라미터
        self.max_acceleration = 50.0  # deg/s²
        self.max_velocity = 200.0     # deg/s
        
    def filter_motion(self, measurement):
        """자연스러운 모션 필터링"""
        # 1. 적응형 칼만 필터 적용
        self.kalman.predict()
        self.kalman.update(measurement)
        
        filtered_angle = self.kalman.get_angle()
        filtered_velocity = self.kalman.get_velocity()
        
        # 2. 모션 예측기 업데이트
        self.predictor.update(filtered_angle, filtered_velocity)
        predicted_angle = self.predictor.predict_next_angle()
        
        # 3. 예측과 필터링 결과 융합
        if predicted_angle is not None:
            # 예측 가중치를 모션 강도에 따라 조절
            motion_intensity = self.kalman.detect_motion_intensity(measurement)
            prediction_weight = min(0.5, motion_intensity * 0.2)
            
            final_angle = (1 - prediction_weight) * filtered_angle + prediction_weight * predicted_angle
        else:
            final_angle = filtered_angle
        
        # 4. 물리적 제약 적용 (자연스러운 움직임)
        final_angle = self.apply_physical_constraints(final_angle, filtered_velocity)
        
        return final_angle, filtered_velocity
    
    def apply_physical_constraints(self, angle, velocity):
        """물리적 제약 적용"""
        # 속도 제한
        if abs(velocity) > self.max_velocity:
            velocity = np.sign(velocity) * self.max_velocity
        
        # 가속도 제한 (각도 변화율 제한)
        angle_change = velocity * self.dt
        max_angle_change = self.max_acceleration * self.dt * self.dt
        
        if abs(angle_change) > max_angle_change:
            angle_change = np.sign(angle_change) * max_angle_change
        
        return angle

# --- 데이터 구조 ---
history_len = 100
imu_data = {name: {k: [0]*history_len for k in ['Roll','Pitch','Yaw']} for name in DEVICES}
data_lock = threading.Lock()
offsets = {name: {'Roll':0,'Pitch':0,'Yaw':0} for name in DEVICES}
calibrated = {name: False for name in DEVICES}
calibration_data = {name: [] for name in DEVICES}
structured_data = {name: [] for name in DEVICES}

# --- 필터 초기화 ---
dt = 0.05
filters = {}
for device_name in DEVICES:
    filters[device_name] = {
        'Roll': NaturalMotionFilter(dt),
        'Pitch': NaturalMotionFilter(dt),
        'Yaw': NaturalMotionFilter(dt)
    }

def calculate_orientation_running_optimized(device_name, ax, ay, az, gx, gy, gz, mx, my):
    """러닝 모션 최적화 방향 계산"""
    # 가속도계 기반 각도 계산
    acc_roll = np.degrees(np.arctan2(ay, az))
    acc_pitch = np.degrees(np.arctan2(-ax, math.hypot(ay, az)))
    mag_yaw = np.degrees(np.arctan2(my, mx))
    
    # 각 축별 자연스러운 모션 필터링
    filtered_angles = {}
    for axis, acc_angle in zip(['Roll', 'Pitch', 'Yaw'], [acc_roll, acc_pitch, mag_yaw]):
        filtered_angle, velocity = filters[device_name][axis].filter_motion(acc_angle)
        filtered_angles[axis] = filtered_angle
    
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

        # 러닝 모션 최적화 방향 계산
        roll, pitch, yaw = calculate_orientation_running_optimized(device_name, *acc, *gyr, *mag)
        
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
                    "filter_type": "running_optimized"
                })
                
                # UDP 전송 (Unity용)
                send_udp({
                    "device": device_name,
                    "roll": round(roll, 2),
                    "pitch": round(pitch, 2),
                    "yaw": round(yaw, 2),
                    "filter_type": "running_optimized"
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
root.title("IMU Running Motion Optimization")

# 한글 폰트 설정
import matplotlib.font_manager as fm
def set_korean_font():
    korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Batang', 'Dotum', 'Gulim']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 폰트 설정 완료: {font}")
            return True
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    return False

set_korean_font()

# 4개 장치에 맞는 그래프 크기 조정
num_devices = len(DEVICES)
fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4행 3열로 변경 (4개 장치 x 3개 축)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

lines = {}
device_list = list(DEVICES.keys())
for i, device in enumerate(device_list):
    for j, key in enumerate(['Roll','Pitch','Yaw']):
        ax = axes[i, j]  # 각 장치가 한 행을 차지
        ax.set_title(f"{device} - {key}")
        ax.set_ylim(-180, 180)
        ax.set_xlim(0, history_len)
        lines[(device, key)], = ax.plot(range(history_len), imu_data[device][key])
        ax.grid(True, alpha=0.3)

def update_graph():
    with data_lock:
        for device in DEVICES:
            for key in ['Roll','Pitch','Yaw']:
                lines[(device, key)].set_ydata(imu_data[device][key])
    canvas.draw_idle()
    root.after(100, update_graph)

def on_close():
    filename = f"running_motion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    print(f"✅ 러닝 모션 JSON 저장 완료: {filename}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

for name, addr in DEVICES.items():
    start_ble_thread(name, addr)

root.after(100, update_graph)
root.mainloop() 