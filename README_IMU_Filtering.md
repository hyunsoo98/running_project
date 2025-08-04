# IMU 노이즈 필터링 가이드

## 📋 개요

이 프로젝트는 IMU(Inertial Measurement Unit) 센서의 노이즈를 효과적으로 제거하기 위한 다양한 필터링 방법들을 제공합니다.

## 🎯 제공되는 필터링 방법

### 1. 기본 필터들
- **이동평균 필터 (Moving Average)**: 간단하고 빠른 노이즈 제거
- **저주파 필터 (Low Pass Filter)**: 고주파 노이즈 제거
- **중간값 필터 (Median Filter)**: 이상치 제거에 효과적
- **버터워스 필터 (Butterworth Filter)**: 주파수 도메인에서의 정밀한 필터링

### 2. 고급 필터들
- **칼만 필터 (Kalman Filter)**: 최적 추정을 통한 노이즈 제거
- **적응형 필터 (Adaptive Filter)**: 노이즈 레벨에 따른 자동 조절
- **상보 필터 (Complementary Filter)**: 가속도계와 자이로스코프 융합
- **Madgwick 필터**: 쿼터니언 기반 고정밀 방향 추정

### 3. 전처리 도구
- **이상치 탐지 (Outlier Detection)**: 급격한 변화나 오류 데이터 제거

## 🚀 설치 및 사용법

### 필요한 패키지 설치
```bash
pip install numpy scipy matplotlib bleak
```

### 기본 사용법

```python
from imu_noise_filtering import IMUNoiseFilter

# 필터 초기화
imu_filter = IMUNoiseFilter(
    window_size=10,      # 이동평균 윈도우 크기
    alpha=0.1,           # 저주파 필터 알파값
    cutoff_freq=5.0,     # 버터워스 필터 차단 주파수 (Hz)
    sample_rate=20.0     # 샘플링 주파수 (Hz)
)

# 노이즈가 있는 IMU 데이터
noisy_roll = 45.0 + np.random.normal(0, 2.0)

# 다양한 필터 적용
ma_filtered = imu_filter.moving_average_filter(noisy_roll, 'roll')
lpf_filtered = imu_filter.low_pass_filter(noisy_roll, 'roll')
kalman_filtered = imu_filter.kalman_filter(noisy_roll, 'roll')
```

## 📊 성능 비교

### 필터별 특징

| 필터 | 장점 | 단점 | 적합한 상황 |
|------|------|------|-------------|
| 이동평균 | 빠름, 간단 | 지연 발생 | 실시간성 중요 |
| 저주파 | 빠름, 안정적 | 고주파 신호 손실 | 일반적인 노이즈 |
| 중간값 | 이상치 제거 | 계산량 많음 | 급격한 변화 |
| 버터워스 | 정밀한 주파수 제거 | 복잡함 | 주파수 분석 필요 |
| 칼만 | 최적 추정 | 복잡함, 튜닝 필요 | 고정밀 요구 |
| 적응형 | 자동 조절 | 복잡함 | 다양한 노이즈 환경 |

### 성능 지표
- **RMSE (Root Mean Square Error)**: 전체적인 오차 측정
- **MAE (Mean Absolute Error)**: 절대 오차 측정
- **처리 시간**: 실시간성 평가

## 🔧 고급 사용법

### 1. 다중 필터 조합
```python
# 단계별 필터링
clean_angle = imu_filter.outlier_detection(noisy_angle, 'roll')
complementary_angle = imu_filter.complementary_filter(clean_angle, gyro_rate, dt)
final_angle = imu_filter.kalman_filter(complementary_angle, 'roll')
```

### 2. 적응형 필터링
```python
# 속도에 따른 필터 강도 조절
filtered_angle = imu_filter.adaptive_filter(angle, 'roll', velocity=angular_velocity)
```

### 3. Madgwick 필터 사용
```python
# 6축 IMU 데이터
gyro = [gx, gy, gz]  # rad/s
accel = [ax, ay, az]  # g

# 9축 IMU 데이터 (자기계 포함)
mag = [mx, my, mz]  # optional

roll, pitch, yaw = imu_filter.madgwick_filter(gyro, accel, mag)
```

## 📈 성능 테스트

### 테스트 실행
```bash
python filter_comparison_test.py
```

### 테스트 결과 해석
1. **RMSE**: 낮을수록 좋음 (정확도)
2. **MAE**: 낮을수록 좋음 (정밀도)
3. **처리 시간**: 낮을수록 좋음 (실시간성)

## 🎛️ 파라미터 튜닝 가이드

### 이동평균 필터
- `window_size`: 클수록 더 부드럽지만 지연 증가
- **권장값**: 5-20

### 저주파 필터
- `alpha`: 작을수록 더 부드럽지만 지연 증가
- **권장값**: 0.05-0.3

### 칼만 필터
- `process_noise`: 클수록 측정값 신뢰도 증가
- `measurement_noise`: 클수록 모델 신뢰도 증가
- **권장값**: process_noise=0.01, measurement_noise=1.0

### 버터워스 필터
- `cutoff_freq`: 샘플링 주파수의 1/4 이하 권장
- **권장값**: sample_rate/4

## 🔍 문제 해결

### 일반적인 문제들

1. **지연이 너무 큼**
   - 윈도우 크기 줄이기
   - 알파값 증가
   - 차단 주파수 증가

2. **노이즈가 여전히 많음**
   - 윈도우 크기 증가
   - 알파값 감소
   - 칼만 필터 사용

3. **급격한 변화 감지 못함**
   - 중간값 필터 사용
   - 적응형 필터 사용
   - 이상치 탐지 비활성화

### 디버깅 팁
```python
# 필터 상태 확인
print(f"이동평균 버퍼 크기: {len(imu_filter.ma_buffers['roll'])}")
print(f"저주파 이전값: {imu_filter.lpf_prev['roll']}")

# 노이즈 레벨 추정
noise_level = np.std(recent_data)
print(f"추정 노이즈 레벨: {noise_level}")
```

## 📝 실제 프로젝트 적용

### 기존 코드 개선
기존 `IOT_imu_multidevice_unity.py`를 `improved_imu_filtering.py`로 교체하여 더 나은 필터링 성능을 얻을 수 있습니다.

### 주요 개선사항
1. **다중 필터 조합**: 이상치 탐지 → 상보 필터 → 적응형 필터 → 칼만 필터
2. **3차 칼만 필터**: 각도, 각속도, 각가속도 모델링
3. **적응형 파라미터**: 노이즈 레벨에 따른 자동 조절
4. **향상된 오프셋 제거**: 더 정확한 캘리브레이션

## 🤝 기여하기

새로운 필터링 방법이나 개선사항을 제안하시면 언제든 환영합니다!

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 