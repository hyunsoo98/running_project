import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from dtw import dtw

# --- [Windows용 한글 폰트 설정] ---
# Windows에 기본으로 설치된 '맑은 고딕'을 사용합니다.
try:
    plt.rc('font', family='Malgun Gothic')
    # 마이너스 기호가 깨지는 것을 방지
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Malgun Gothic 폰트를 찾을 수 없습니다. Windows 환경이 맞는지 확인해주세요.")
# ------------------------------------

# --- 1. 데이터 로드 및 보행 주기 분할 ---
try:
    df = pd.read_csv('gait_analysis_with_coords.csv')
except FileNotFoundError:
    print("오류: 'gait_analysis_with_coords.csv' 파일을 찾을 수 없습니다.")
    print("먼저 영상 처리 스크립트를 실행하여 이 파일을 생성해주세요.")
    exit()

# 왼쪽 발의 Heel Strike 지점 찾기
peaks_left, _ = find_peaks(df['Left_Ankle_y'], height=np.mean(df['Left_Ankle_y']), distance=20)

if len(peaks_left) < 2:
    print("DTW 분석을 수행하기에 보행 주기(gait cycles)가 충분하지 않습니다.")
    exit()

# 각 보행 주기의 무릎 각도 데이터를 리스트에 저장
left_knee_cycles = []
for i in range(len(peaks_left) - 1):
    start_idx = peaks_left[i]
    end_idx = peaks_left[i+1]
    cycle_data = df['Left_Knee_Angle'][start_idx:end_idx].values
    left_knee_cycles.append(cycle_data)

print(f"총 {len(left_knee_cycles)}개의 왼쪽 보행 주기가 감지되었습니다.")

# --- 2. 템플릿(기준) 패턴 설정 ---
template_cycle = left_knee_cycles[0]

# --- 3. DTW 유사도 계산 ---
dtw_distances = []
print("\n--- DTW 분석 결과 (템플릿과의 거리) ---")

for i, cycle in enumerate(left_knee_cycles):
    alignment = dtw(template_cycle, cycle, keep_internals=True, 
                    step_pattern='symmetric2')
    
    dtw_distances.append(alignment.normalizedDistance)
    print(f"Cycle {i+1} vs Template: DTW 거리 = {alignment.normalizedDistance:.4f}")

# --- 4. 시각화 ---
if len(dtw_distances) > 1:
    max_dist_index = np.argmax(dtw_distances[1:]) + 1 
    max_dist_cycle = left_knee_cycles[max_dist_index]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(template_cycle, label=f'템플릿 사이클 (Cycle 1)', color='blue')
    ax1.plot(max_dist_cycle, label=f'가장 다른 사이클 (Cycle {max_dist_index + 1})', color='red')
    ax1.set_title('템플릿과 가장 다른 사이클의 패턴 비교')
    ax1.set_xlabel('프레임')
    ax1.set_ylabel('왼쪽 무릎 각도 (도)')
    ax1.legend()
    ax1.grid(True)
    
    alignment_to_plot = dtw(template_cycle, max_dist_cycle, keep_internals=True)
    plt.sca(ax2)
    alignment_to_plot.plot(type="twoway", offset=-1)
    ax2.set_title(f'DTW 양방향 정렬 경로 (템플릿 vs Cycle {max_dist_index + 1})')
    
    plt.tight_layout()
    plt.show()

# DTW 거리 결과 요약 막대그래프
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(dtw_distances) + 1), dtw_distances)
plt.title('각 보행 주기와 템플릿 간의 DTW 거리')
plt.xlabel('보행 주기 번호')
plt.ylabel('정규화된 DTW 거리')
plt.xticks(range(1, len(dtw_distances) + 1))
plt.grid(axis='y')
plt.show()
