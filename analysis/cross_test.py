import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from dtw import dtw

# --- 1. 한글 폰트 설정 ---
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Malgun Gothic 폰트를 찾을 수 없습니다. Windows 환경이 맞는지 확인해주세요.")

# --- 2. 데이터 로드 ---
try:
    df_video = pd.read_csv('gait_analysis_with_coords.csv')
    df_imu = pd.read_csv('imu_data5.csv')
except FileNotFoundError as e:
    print(f"오류: {e.filename} 파일을 찾을 수 없습니다.")
    exit()

# --- 3. IMU 데이터 시간 처리 ---
df_imu['time'] = pd.to_datetime(df_imu['time'], format='%H:%M:%S.%f').dt.time
start_time_imu = df_imu['time'].iloc[0]
df_imu['Time_sec'] = df_imu['time'].apply(
    lambda t: (t.hour - start_time_imu.hour) * 3600 + 
              (t.minute - start_time_imu.minute) * 60 + 
              (t.second - start_time_imu.second) + 
              (t.microsecond - start_time_imu.microsecond) / 1e6
)

# --- 4. 보행 주기별 통합 데이터 생성 ---
peaks_left, _ = find_peaks(df_video['Left_Ankle_y'], height=np.mean(df_video['Left_Ankle_y']), distance=20)
if len(peaks_left) < 2:
    print("분석할 보행 주기가 충분하지 않습니다.")
    exit()

gait_cycles_analysis = []
template_cycle_data = df_video['Left_Knee_Angle'][peaks_left[0]:peaks_left[1]].values

for i in range(len(peaks_left) - 1):
    # 영상 데이터에서 주기 정보 추출
    start_idx_video = peaks_left[i]
    end_idx_video = peaks_left[i+1]
    start_time = df_video['Time_sec'][start_idx_video]
    end_time = df_video['Time_sec'][end_idx_video]
    
    # DTW 거리 계산 (영상 기반 이상 점수)
    current_cycle_data = df_video['Left_Knee_Angle'][start_idx_video:end_idx_video].values
    alignment = dtw(template_cycle_data, current_cycle_data, keep_internals=True, step_pattern='symmetric2')
    dtw_dist = alignment.normalizedDistance
    
    # IMU 데이터에서 해당 시간 구간의 데이터 추출
    imu_cycle_data = df_imu[(df_imu['Time_sec'] >= start_time) & (df_imu['Time_sec'] <= end_time)]
    
    # IMU 데이터로 불안정성 점수 계산 (표준편차)
    if not imu_cycle_data.empty:
        yaw_std = imu_cycle_data['Yaw'].std()
        roll_std = imu_cycle_data['Roll'].std()
    else:
        yaw_std, roll_std = 0, 0
        
    gait_cycles_analysis.append({
        'cycle_num': i + 1,
        'dtw_distance': dtw_dist,
        'yaw_std': yaw_std,
        'roll_std': roll_std
    })

df_analysis = pd.DataFrame(gait_cycles_analysis)
print("보행 주기별 통합 분석 완료.")

# --- 5. 개선된 통합 시각화 ---
fig, ax1 = plt.subplots(figsize=(18, 8))
fig.suptitle('보행 주기별 이상 점수 교차 검증', fontsize=20)

# 1번 축 (왼쪽): 영상 기반 DTW 거리 (막대 그래프)
color = 'tab:blue'
ax1.set_xlabel('보행 주기 번호', fontsize=14)
ax1.set_ylabel('DTW 거리 (패턴 이상 점수)', color=color, fontsize=12)
ax1.bar(df_analysis['cycle_num'], df_analysis['dtw_distance'], color=color, alpha=0.6, label='DTW 거리 (영상)')
ax1.tick_params(axis='y', labelcolor=color)

# 2번 축 (오른쪽): IMU 기반 불안정성 (선 그래프)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('IMU 표준편차 (물리적 불안정성)', color=color, fontsize=12)
ax2.plot(df_analysis['cycle_num'], df_analysis['roll_std'], color='red', marker='o', linestyle='--', label='Roll 표준편차 (IMU)')
ax2.plot(df_analysis['cycle_num'], df_analysis['yaw_std'], color='green', marker='x', linestyle=':', label='Yaw 표준편차 (IMU)')
ax2.tick_params(axis='y', labelcolor=color)

# 범례 합치기
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
