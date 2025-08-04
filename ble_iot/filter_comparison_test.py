import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
import time
from imu_noise_filtering import IMUNoiseFilter, KalmanFilter, MadgwickFilter

def generate_synthetic_imu_data(duration=10.0, sample_rate=20.0, noise_level=2.0):
    """합성 IMU 데이터 생성"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # 실제 신호 (사인파 + 계단 함수)
    true_roll = 30 * np.sin(2 * np.pi * 0.5 * t) + 10 * np.heaviside(t - 5, 0)
    true_pitch = 20 * np.cos(2 * np.pi * 0.3 * t) + 15 * np.heaviside(t - 3, 0)
    true_yaw = 45 * np.sin(2 * np.pi * 0.2 * t)
    
    # 노이즈 추가
    noisy_roll = true_roll + np.random.normal(0, noise_level, len(t))
    noisy_pitch = true_pitch + np.random.normal(0, noise_level, len(t))
    noisy_yaw = true_yaw + np.random.normal(0, noise_level, len(t))
    
    # 가속도계 데이터 (중력 벡터 기반)
    accel_x = np.sin(np.radians(noisy_roll)) * np.cos(np.radians(noisy_pitch))
    accel_y = np.sin(np.radians(noisy_pitch))
    accel_z = np.cos(np.radians(noisy_roll)) * np.cos(np.radians(noisy_pitch))
    
    # 자이로스코프 데이터 (각속도)
    gyro_x = np.gradient(noisy_roll) * sample_rate / 360  # deg/s
    gyro_y = np.gradient(noisy_pitch) * sample_rate / 360
    gyro_z = np.gradient(noisy_yaw) * sample_rate / 360
    
    return {
        'time': t,
        'true': {'roll': true_roll, 'pitch': true_pitch, 'yaw': true_yaw},
        'noisy': {'roll': noisy_roll, 'pitch': noisy_pitch, 'yaw': noisy_yaw},
        'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
        'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z}
    }

def calculate_rmse(original, filtered):
    """RMSE (Root Mean Square Error) 계산"""
    return np.sqrt(np.mean((original - filtered) ** 2))

def calculate_mae(original, filtered):
    """MAE (Mean Absolute Error) 계산"""
    return np.mean(np.abs(original - filtered))

def test_filters():
    """다양한 필터 성능 테스트"""
    print("🔬 IMU 필터링 성능 비교 테스트 시작...")
    
    # 합성 데이터 생성
    data = generate_synthetic_imu_data(duration=10.0, sample_rate=20.0, noise_level=3.0)
    
    # 필터 초기화
    imu_filter = IMUNoiseFilter(
        window_size=10,
        alpha=0.1,
        cutoff_freq=5.0,
        sample_rate=20.0
    )
    
    # 결과 저장용
    results = {
        'moving_average': {'roll': [], 'pitch': [], 'yaw': []},
        'low_pass': {'roll': [], 'pitch': [], 'yaw': []},
        'median': {'roll': [], 'pitch': [], 'yaw': []},
        'butterworth': {'roll': [], 'pitch': [], 'yaw': []},
        'kalman': {'roll': [], 'pitch': [], 'yaw': []},
        'adaptive': {'roll': [], 'pitch': [], 'yaw': []}
    }
    
    # 각 샘플에 대해 필터 적용
    for i in range(len(data['time'])):
        for axis in ['roll', 'pitch', 'yaw']:
            noisy_value = data['noisy'][axis][i]
            
            # 이동평균 필터
            ma_filtered = imu_filter.moving_average_filter(noisy_value, axis)
            results['moving_average'][axis].append(ma_filtered)
            
            # 저주파 필터
            lpf_filtered = imu_filter.low_pass_filter(noisy_value, axis)
            results['low_pass'][axis].append(lpf_filtered)
            
            # 중간값 필터
            median_filtered = imu_filter.median_filter(noisy_value, axis)
            results['median'][axis].append(median_filtered)
            
            # 버터워스 필터
            butterworth_filtered = imu_filter.butterworth_filter(noisy_value, axis)
            results['butterworth'][axis].append(butterworth_filtered)
            
            # 칼만 필터
            kalman_filtered = imu_filter.kalman_filter(noisy_value, axis)
            results['kalman'][axis].append(kalman_filtered)
            
            # 적응형 필터
            adaptive_filtered = imu_filter.adaptive_filter(noisy_value, axis)
            results['adaptive'][axis].append(adaptive_filtered)
    
    # 성능 평가
    print("\n📊 filter_comparison:")
    print("=" * 60)
    
    for axis in ['roll', 'pitch', 'yaw']:
        print(f"\n{axis.upper()} axis:")
        print("-" * 30)
        
        true_signal = data['true'][axis]
        
        for filter_name in results.keys():
            filtered_signal = np.array(results[filter_name][axis])
            
            rmse_val = calculate_rmse(true_signal, filtered_signal)
            mae_val = calculate_mae(true_signal, filtered_signal)
            
            print(f"{filter_name:15s}: RMSE={rmse_val:6.3f}, MAE={mae_val:6.3f}")
    
    return data, results

def plot_comparison(data, results):
    """filtering visualization"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('IMU filtering comparison', fontsize=16)
    
    colors = {
        'true': 'black',
        'noisy': 'red',
        'moving_average': 'blue',
        'low_pass': 'green',
        'median': 'orange',
        'butterworth': 'purple',
        'kalman': 'brown',
        'adaptive': 'pink'
    }
    
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        # 첫 번째 열: 원본 신호와 노이즈
        ax1 = axes[i, 0]
        ax1.plot(data['time'], data['true'][axis], color=colors['true'], 
                label='True Signal', linewidth=2)
        ax1.plot(data['time'], data['noisy'][axis], color=colors['noisy'], 
                label='Noisy Signal', alpha=0.7)
        ax1.set_title(f'{axis.upper()} - law vs noise')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angle (degrees)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 두 번째 열: 필터링 결과
        ax2 = axes[i, 1]
        ax2.plot(data['time'], data['true'][axis], color=colors['true'], 
                label='True Signal', linewidth=2)
        
        for filter_name in ['kalman', 'adaptive', 'butterworth']:
            filtered_signal = np.array(results[filter_name][axis])
            ax2.plot(data['time'], filtered_signal, color=colors[filter_name], 
                    label=f'{filter_name.title()} Filter', alpha=0.8)
        
        ax2.set_title(f'{axis.upper()} - filtering result')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (degrees)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_real_time_performance():
    """real-time performance test"""
    print("\n⚡ real-time performance test...")
    
    imu_filter = IMUNoiseFilter()
    
    # 각 필터별 처리 시간 측정
    test_data = np.random.normal(0, 2.0, 1000)
    
    filter_methods = [
        ('Moving Average', lambda x: imu_filter.moving_average_filter(x, 'roll')),
        ('Low Pass', lambda x: imu_filter.low_pass_filter(x, 'roll')),
        ('Median', lambda x: imu_filter.median_filter(x, 'roll')),
        ('Butterworth', lambda x: imu_filter.butterworth_filter(x, 'roll')),
        ('Kalman', lambda x: imu_filter.kalman_filter(x, 'roll')),
        ('Adaptive', lambda x: imu_filter.adaptive_filter(x, 'roll'))
    ]
    
    print("\n처리 시간 비교 (1000 샘플):")
    print("-" * 40)
    
    for method_name, filter_func in filter_methods:
        start_time = time.time()
        
        for value in test_data:
            filtered = filter_func(value)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms
        
        print(f"{method_name:15s}: {processing_time:8.2f} ms")

def test_noise_robustness():
    """noise robustness test"""
    print("\n🛡️ noise robustness test...")
    
    noise_levels = [0.5, 1.0, 2.0, 3.0, 5.0]
    imu_filter = IMUNoiseFilter()
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"\n노이즈 레벨: {noise_level}")
        print("-" * 25)
        
        data = generate_synthetic_imu_data(noise_level=noise_level)
        
        # 칼만 필터만 테스트 (대표적)
        kalman_results = []
        for i in range(len(data['time'])):
            filtered = imu_filter.kalman_filter(data['noisy']['roll'][i], 'roll')
            kalman_results.append(filtered)
        
        rmse_val = calculate_rmse(data['true']['roll'], np.array(kalman_results))
        print(f"Kalman Filter RMSE: {rmse_val:.3f}")

def main():
    """main test function"""
    print("🚀 IMU noise filtering comprehensive test")
    print("=" * 50)
    
    # 1. 기본 필터 성능 비교
    data, results = test_filters()
    
    # 2. 시각화
    plot_comparison(data, results)
    
    # 3. 실시간 성능 테스트
    test_real_time_performance()
    
    # 4. 노이즈 강건성 테스트
    test_noise_robustness()
    
    print("\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    main() 