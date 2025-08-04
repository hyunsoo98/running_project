import numpy as np
import math
from scipy import signal
from collections import deque
import time

class IMUNoiseFilter:
    """IMU 노이즈 필터링을 위한 다양한 필터 클래스"""
    
    def __init__(self, window_size=10, alpha=0.1, cutoff_freq=5.0, sample_rate=20.0):
        """
        Args:
            window_size: 이동평균 필터 윈도우 크기
            alpha: 저주파 필터 알파값 (0~1)
            cutoff_freq: 버터워스 필터 차단 주파수 (Hz)
            sample_rate: 샘플링 주파수 (Hz)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        
        # 필터 초기화
        self._init_filters()
        
    def _init_filters(self):
        """다양한 필터 초기화"""
        # 이동평균 필터용 버퍼
        self.ma_buffers = {
            'roll': deque(maxlen=self.window_size),
            'pitch': deque(maxlen=self.window_size),
            'yaw': deque(maxlen=self.window_size)
        }
        
        # 저주파 필터용 이전값
        self.lpf_prev = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }
        
        # 중간값 필터용 버퍼
        self.median_buffers = {
            'roll': deque(maxlen=5),
            'pitch': deque(maxlen=5),
            'yaw': deque(maxlen=5)
        }
        
        # 버터워스 필터 설계
        self._design_butterworth_filter()
        
        # 칼만 필터 초기화
        self.kalman_filters = {
            'roll': KalmanFilter(dt=1.0/self.sample_rate),
            'pitch': KalmanFilter(dt=1.0/self.sample_rate),
            'yaw': KalmanFilter(dt=1.0/self.sample_rate)
        }
        
        # Madgwick 필터 초기화
        self.madgwick_filter = MadgwickFilter(sample_rate=self.sample_rate)
        
    def _design_butterworth_filter(self):
        """버터워스 저역통과 필터 설계"""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = signal.butter(4, normalized_cutoff, btype='low')
        
        # 필터 상태 저장용
        self.butterworth_states = {
            'roll': signal.lfilter_zi(self.b, self.a),
            'pitch': signal.lfilter_zi(self.b, self.a),
            'yaw': signal.lfilter_zi(self.b, self.a)
        }

class KalmanFilter:
    """개선된 칼만 필터"""
    
    def __init__(self, dt, process_noise=0.01, measurement_noise=1.0):
        """
        Args:
            dt: 샘플링 시간 간격
            process_noise: 프로세스 노이즈 분산
            measurement_noise: 측정 노이즈 분산
        """
        self.dt = dt
        
        # 상태 벡터: [angle, angular_velocity]
        self.x = np.zeros((2, 1))
        
        # 상태 공분산 행렬
        self.P = np.eye(2) * 1000
        
        # 상태 전이 행렬
        self.F = np.array([[1, dt], [0, 1]])
        
        # 측정 행렬
        self.H = np.array([[1, 0]])
        
        # 프로세스 노이즈 공분산
        self.Q = np.eye(2) * process_noise
        
        # 측정 노이즈 공분산
        self.R = np.array([[measurement_noise]])
        
    def predict(self):
        """예측 단계"""
        # 상태 예측
        self.x = self.F @ self.x
        
        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """업데이트 단계"""
        # 칼만 게인 계산
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        
        # 공분산 업데이트
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
    def get_angle(self):
        """현재 각도 반환"""
        return float(self.x[0, 0])
    
    def get_angular_velocity(self):
        """현재 각속도 반환"""
        return float(self.x[1, 0])

class MadgwickFilter:
    """Madgwick 필터 - 자이로스코프와 가속도계 융합"""
    
    def __init__(self, sample_rate, beta=0.1):
        """
        Args:
            sample_rate: 샘플링 주파수
            beta: 필터 게인 (0~1)
        """
        self.sample_rate = sample_rate
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 쿼터니언
        
    def update(self, gyro, accel, mag=None):
        """
        Args:
            gyro: 자이로스코프 데이터 [x, y, z] (rad/s)
            accel: 가속도계 데이터 [x, y, z] (g)
            mag: 자기계 데이터 [x, y, z] (optional)
        """
        # 정규화
        accel = accel / np.linalg.norm(accel)
        if mag is not None:
            mag = mag / np.linalg.norm(mag)
        
        # 쿼터니언 업데이트
        self._update_quaternion(gyro, accel, mag)
        
        # 오일러 각도로 변환
        return self._quaternion_to_euler()
    
    def _update_quaternion(self, gyro, accel, mag=None):
        """쿼터니언 업데이트"""
        dt = 1.0 / self.sample_rate
        
        # 자이로스코프 적분
        q_dot = 0.5 * self._quaternion_multiply(self.q, np.array([0, *gyro]))
        
        # 가속도계 보정
        if mag is not None:
            # 9축 융합
            q_accel = self._accel_mag_to_quaternion(accel, mag)
        else:
            # 6축 융합
            q_accel = self._accel_to_quaternion(accel)
        
        # Madgwick 알고리즘
        q_correction = self.beta * (q_accel - self.q)
        self.q = self.q + (q_dot + q_correction) * dt
        self.q = self.q / np.linalg.norm(self.q)
    
    def _quaternion_multiply(self, q1, q2):
        """쿼터니언 곱셈"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _accel_to_quaternion(self, accel):
        """가속도계 데이터를 쿼터니언으로 변환"""
        ax, ay, az = accel
        
        # 중력 방향 추정
        if az >= 0:
            q = np.array([np.sqrt((az + 1) / 2), -ay / (2 * np.sqrt((az + 1) / 2)), 
                         ax / (2 * np.sqrt((az + 1) / 2)), 0])
        else:
            q = np.array([-ay / (2 * np.sqrt((1 - az) / 2)), np.sqrt((1 - az) / 2), 
                         0, ax / (2 * np.sqrt((1 - az) / 2))])
        
        return q / np.linalg.norm(q)
    
    def _accel_mag_to_quaternion(self, accel, mag):
        """가속도계와 자기계 데이터를 쿼터니언으로 변환"""
        # 간단한 구현 - 실제로는 더 복잡한 알고리즘 필요
        return self._accel_to_quaternion(accel)
    
    def _quaternion_to_euler(self):
        """쿼터니언을 오일러 각도로 변환"""
        w, x, y, z = self.q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.degrees([roll, pitch, yaw])

class IMUNoiseFilter:
    """IMU 노이즈 필터링을 위한 다양한 필터 클래스"""
    
    def __init__(self, window_size=10, alpha=0.1, cutoff_freq=5.0, sample_rate=20.0):
        """
        Args:
            window_size: 이동평균 필터 윈도우 크기
            alpha: 저주파 필터 알파값 (0~1)
            cutoff_freq: 버터워스 필터 차단 주파수 (Hz)
            sample_rate: 샘플링 주파수 (Hz)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        
        # 필터 초기화
        self._init_filters()
        
    def _init_filters(self):
        """다양한 필터 초기화"""
        # 이동평균 필터용 버퍼
        self.ma_buffers = {
            'roll': deque(maxlen=self.window_size),
            'pitch': deque(maxlen=self.window_size),
            'yaw': deque(maxlen=self.window_size)
        }
        
        # 저주파 필터용 이전값
        self.lpf_prev = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }
        
        # 중간값 필터용 버퍼
        self.median_buffers = {
            'roll': deque(maxlen=5),
            'pitch': deque(maxlen=5),
            'yaw': deque(maxlen=5)
        }
        
        # 버터워스 필터 설계
        self._design_butterworth_filter()
        
        # 칼만 필터 초기화
        self.kalman_filters = {
            'roll': KalmanFilter(dt=1.0/self.sample_rate),
            'pitch': KalmanFilter(dt=1.0/self.sample_rate),
            'yaw': KalmanFilter(dt=1.0/self.sample_rate)
        }
        
        # Madgwick 필터 초기화
        self.madgwick_filter = MadgwickFilter(sample_rate=self.sample_rate)
        
    def _design_butterworth_filter(self):
        """버터워스 저역통과 필터 설계"""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = signal.butter(4, normalized_cutoff, btype='low')
        
        # 필터 상태 저장용
        self.butterworth_states = {
            'roll': signal.lfilter_zi(self.b, self.a),
            'pitch': signal.lfilter_zi(self.b, self.a),
            'yaw': signal.lfilter_zi(self.b, self.a)
        }
    
    def moving_average_filter(self, angle, axis):
        """이동평균 필터"""
        self.ma_buffers[axis].append(angle)
        if len(self.ma_buffers[axis]) > 0:
            return np.mean(self.ma_buffers[axis])
        return angle
    
    def low_pass_filter(self, angle, axis):
        """저주파 필터 (1차 IIR)"""
        filtered = self.alpha * angle + (1 - self.alpha) * self.lpf_prev[axis]
        self.lpf_prev[axis] = filtered
        return filtered
    
    def median_filter(self, angle, axis):
        """중간값 필터"""
        self.median_buffers[axis].append(angle)
        if len(self.median_buffers[axis]) > 0:
            return np.median(self.median_buffers[axis])
        return angle
    
    def butterworth_filter(self, angle, axis):
        """버터워스 저역통과 필터"""
        filtered, self.butterworth_states[axis] = signal.lfilter(
            self.b, self.a, [angle], zi=self.butterworth_states[axis]
        )
        return filtered[0]
    
    def kalman_filter(self, angle, axis):
        """칼만 필터"""
        self.kalman_filters[axis].predict()
        self.kalman_filters[axis].update(np.array([[angle]]))
        return self.kalman_filters[axis].get_angle()
    
    def madgwick_filter(self, gyro, accel, mag=None):
        """Madgwick 필터"""
        return self.madgwick_filter.update(gyro, accel, mag)
    
    def adaptive_filter(self, angle, axis, velocity=None):
        """적응형 필터 - 속도에 따라 필터 강도 조절"""
        if velocity is None:
            # 속도 정보가 없으면 기본 필터 사용
            return self.low_pass_filter(angle, axis)
        
        # 속도에 따른 알파값 조절
        velocity_magnitude = abs(velocity)
        if velocity_magnitude > 10:  # 빠른 움직임
            adaptive_alpha = 0.8  # 적은 필터링
        elif velocity_magnitude > 5:  # 중간 움직임
            adaptive_alpha = 0.5
        else:  # 느린 움직임
            adaptive_alpha = 0.1  # 강한 필터링
        
        filtered = adaptive_alpha * angle + (1 - adaptive_alpha) * self.lpf_prev[axis]
        self.lpf_prev[axis] = filtered
        return filtered
    
    def complementary_filter(self, accel_angle, gyro_rate, axis, dt):
        """상보 필터 - 가속도계와 자이로스코프 융합"""
        alpha = 0.98  # 자이로스코프 신뢰도
        
        # 자이로스코프 적분
        gyro_angle = self.lpf_prev[axis] + gyro_rate * dt
        
        # 상보 필터 적용
        filtered = alpha * gyro_angle + (1 - alpha) * accel_angle
        self.lpf_prev[axis] = filtered
        return filtered
    
    def outlier_detection(self, angle, axis, threshold=3.0):
        """이상치 탐지 및 제거"""
        if len(self.ma_buffers[axis]) < 3:
            return angle
        
        mean_val = np.mean(self.ma_buffers[axis])
        std_val = np.std(self.ma_buffers[axis])
        
        if abs(angle - mean_val) > threshold * std_val:
            # 이상치인 경우 이전값 반환
            return self.lpf_prev[axis]
        
        return angle

# 사용 예시
def example_usage():
    """필터 사용 예시"""
    # 필터 초기화
    imu_filter = IMUNoiseFilter(
        window_size=10,
        alpha=0.1,
        cutoff_freq=5.0,
        sample_rate=20.0
    )
    
    # 시뮬레이션 데이터
    noisy_roll = 45.0 + np.random.normal(0, 2.0)  # 노이즈가 있는 롤 각도
    
    # 다양한 필터 적용
    ma_filtered = imu_filter.moving_average_filter(noisy_roll, 'roll')
    lpf_filtered = imu_filter.low_pass_filter(noisy_roll, 'roll')
    median_filtered = imu_filter.median_filter(noisy_roll, 'roll')
    butterworth_filtered = imu_filter.butterworth_filter(noisy_roll, 'roll')
    kalman_filtered = imu_filter.kalman_filter(noisy_roll, 'roll')
    
    print(f"원본: {noisy_roll:.2f}")
    print(f"이동평균: {ma_filtered:.2f}")
    print(f"저주파: {lpf_filtered:.2f}")
    print(f"중간값: {median_filtered:.2f}")
    print(f"버터워스: {butterworth_filtered:.2f}")
    print(f"칼만: {kalman_filtered:.2f}")

if __name__ == "__main__":
    example_usage() 