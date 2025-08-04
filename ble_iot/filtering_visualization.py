import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
import math

# 한글 폰트 설정
import matplotlib.font_manager as fm

# 윈도우에서 사용 가능한 한글 폰트 찾기
def set_korean_font():
    """한글 폰트 설정 함수"""
    # 윈도우 기본 한글 폰트들
    korean_fonts = [
        'Malgun Gothic',      # 윈도우 10/11 기본
        'NanumGothic',        # 나눔고딕
        'NanumBarunGothic',   # 나눔바른고딕
        'Batang',             # 바탕
        'Dotum',              # 돋움
        'Gulim'               # 굴림
    ]
    
    # 시스템에 설치된 폰트 확인
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 사용 가능한 한글 폰트 찾기
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 폰트 설정 완료: {font}")
            return True
    
    # 한글 폰트가 없으면 기본 폰트 사용
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    return False

# 한글 폰트 설정 적용
set_korean_font() 

# --- 필터링 클래스들 (기존 코드에서 가져옴) ---
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

# --- 데이터 로드 및 전처리 ---
def load_and_preprocess_data():
    """CSV 데이터 로드 및 전처리"""
    # 데이터 로드
    df = pd.read_csv('pigeon_project/imu_data5.csv')
    
    # 시간을 초 단위로 변환
    df['time_seconds'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f').astype(np.int64) // 10**9
    df['time_seconds'] = df['time_seconds'] - df['time_seconds'].iloc[0]
    
    # 샘플링 간격 계산
    dt = np.mean(np.diff(df['time_seconds']))
    
    print(f"데이터 정보:")
    print(f"- 총 데이터 포인트: {len(df)}")
    print(f"- 샘플링 간격: {dt:.3f}초")
    print(f"- 측정 시간: {df['time_seconds'].iloc[-1]:.1f}초")
    
    return df, dt

def apply_filters_to_data(df, dt):
    """각 필터를 데이터에 적용하고 결과 저장"""
    
    # 필터 초기화
    filters = {
        'outlier_detector': OutlierDetector(window_size=10, threshold=3.0),
        'adaptive': AdaptiveFilter(window_size=20, initial_alpha=0.1),
        'kalman': ImprovedKalmanFilter(dt, process_noise=0.01, measurement_noise=1.0),
        'complementary': ComplementaryFilter(alpha=0.98)
    }
    
    # 결과 저장용 딕셔너리
    results = {
        'original': df['Roll'].values,
        'outlier_removed': [],
        'adaptive_filtered': [],
        'kalman_filtered': [],
        'complementary_filtered': [],
        'final_filtered': []
    }
    
    # 각 데이터 포인트에 대해 필터 적용
    for i, roll_value in enumerate(df['Roll'].values):
        # 1. 이상치 탐지 및 제거
        outlier_removed = filters['outlier_detector'].detect_and_remove(roll_value)
        results['outlier_removed'].append(outlier_removed)
        
        # 2. 적응형 필터 (자이로스코프 데이터가 없으므로 가속도계만 사용)
        adaptive_filtered = filters['adaptive'].update(outlier_removed)
        results['adaptive_filtered'].append(adaptive_filtered)
        
        # 3. 칼만 필터
        kalman_filter = filters['kalman']
        kalman_filter.predict()
        kalman_filter.update(np.array([[adaptive_filtered]]))
        kalman_filtered = kalman_filter.get_angle()
        results['kalman_filtered'].append(kalman_filtered)
        
        # 4. 상보 필터 (자이로스코프 데이터가 없으므로 가속도계만 사용)
        # 자이로스코프 각속도는 이전 값과의 차이로 추정
        if i > 0:
            gyro_rate = (roll_value - df['Roll'].iloc[i-1]) / dt
        else:
            gyro_rate = 0.0
        
        complementary_filtered = filters['complementary'].update(roll_value, gyro_rate, dt)
        results['complementary_filtered'].append(complementary_filtered)
        
        # 5. 최종 필터링 (칼만 필터 결과 사용)
        results['final_filtered'].append(kalman_filtered)
    
    return results

def visualize_filtering_process(df, results):
    """필터링 과정을 단계별로 시각화"""
    
    # 한글 폰트 재확인
    set_korean_font()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('IMU 데이터 필터링 과정 시각화', fontsize=16, fontweight='bold')
    
    # 시간 축
    time_axis = df['time_seconds'].values
    
    # 1. 원본 데이터 vs 이상치 제거
    ax1 = axes[0, 0]
    ax1.plot(time_axis, results['original'], 'b-', alpha=0.7, label='원본 데이터', linewidth=1)
    ax1.plot(time_axis, results['outlier_removed'], 'r-', label='이상치 제거', linewidth=2)
    ax1.set_title('1단계: 이상치 탐지 및 제거')
    ax1.set_ylabel('Roll (도)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 적응형 필터
    ax2 = axes[0, 1]
    ax2.plot(time_axis, results['outlier_removed'], 'b-', alpha=0.7, label='이상치 제거 후', linewidth=1)
    ax2.plot(time_axis, results['adaptive_filtered'], 'g-', label='적응형 필터', linewidth=2)
    ax2.set_title('2단계: 적응형 필터링')
    ax2.set_ylabel('Roll (도)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 칼만 필터
    ax3 = axes[1, 0]
    ax3.plot(time_axis, results['adaptive_filtered'], 'b-', alpha=0.7, label='적응형 필터 후', linewidth=1)
    ax3.plot(time_axis, results['kalman_filtered'], 'orange', label='칼만 필터', linewidth=2)
    ax3.set_title('3단계: 칼만 필터링')
    ax3.set_ylabel('Roll (도)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 상보 필터
    ax4 = axes[1, 1]
    ax4.plot(time_axis, results['original'], 'b-', alpha=0.7, label='원본 데이터', linewidth=1)
    ax4.plot(time_axis, results['complementary_filtered'], 'purple', label='상보 필터', linewidth=2)
    ax4.set_title('4단계: 상보 필터링')
    ax4.set_ylabel('Roll (도)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 최종 비교
    ax5 = axes[2, 0]
    ax5.plot(time_axis, results['original'], 'b-', alpha=0.5, label='원본 데이터', linewidth=1)
    ax5.plot(time_axis, results['final_filtered'], 'red', label='최종 필터링', linewidth=2)
    ax5.set_title('5단계: 최종 필터링 결과')
    ax5.set_xlabel('시간 (초)')
    ax5.set_ylabel('Roll (도)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 모든 필터 비교
    ax6 = axes[2, 1]
    ax6.plot(time_axis, results['original'], 'b-', alpha=0.5, label='원본', linewidth=1)
    ax6.plot(time_axis, results['outlier_removed'], 'g-', alpha=0.7, label='이상치 제거', linewidth=1)
    ax6.plot(time_axis, results['adaptive_filtered'], 'orange', alpha=0.7, label='적응형', linewidth=1)
    ax6.plot(time_axis, results['kalman_filtered'], 'red', label='칼만', linewidth=2)
    ax6.plot(time_axis, results['complementary_filtered'], 'purple', alpha=0.7, label='상보', linewidth=1)
    ax6.set_title('6단계: 모든 필터 비교')
    ax6.set_xlabel('시간 (초)')
    ax6.set_ylabel('Roll (도)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_filter_performance(df, results):
    """필터 성능 분석"""
    
    print("\n" + "="*60)
    print("필터 성능 분석")
    print("="*60)
    
    # 노이즈 레벨 계산 (표준편차)
    noise_levels = {}
    for filter_name, data in results.items():
        noise_levels[filter_name] = np.std(data)
    
    print("노이즈 레벨 (표준편차):")
    for filter_name, noise in noise_levels.items():
        print(f"  {filter_name:20s}: {noise:.3f}")
    
    # 노이즈 감소율 계산
    original_noise = noise_levels['original']
    print(f"\n노이즈 감소율 (원본 대비):")
    for filter_name, noise in noise_levels.items():
        if filter_name != 'original':
            reduction = (original_noise - noise) / original_noise * 100
            print(f"  {filter_name:20s}: {reduction:.1f}% 감소")
    
    # 평균 절대 오차 계산 (이동 평균과의 차이)
    window_size = 10
    moving_avg = pd.Series(results['original']).rolling(window=window_size, center=True).mean()
    
    mae_values = {}
    for filter_name, data in results.items():
        if filter_name != 'original':
            mae = np.mean(np.abs(np.array(data) - moving_avg))
            mae_values[filter_name] = mae
    
    print(f"\n평균 절대 오차 (이동평균 대비):")
    for filter_name, mae in mae_values.items():
        print(f"  {filter_name:20s}: {mae:.3f}")

def main():
    """메인 함수"""
    print("IMU 데이터 필터링 과정 시각화")
    print("="*50)
    
    # 1. 데이터 로드
    df, dt = load_and_preprocess_data()
    
    # 2. 필터 적용
    print("\n필터 적용 중...")
    results = apply_filters_to_data(df, dt)
    
    # 3. 성능 분석
    analyze_filter_performance(df, results)
    
    # 4. 시각화
    print("\n시각화 생성 중...")
    visualize_filtering_process(df, results)
    
    print("\n✅ 필터링 과정 시각화 완료!")

if __name__ == "__main__":
    main() 