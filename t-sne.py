import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 1. 한글 폰트 설정 ---
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Malgun Gothic 폰트를 찾을 수 없습니다. Windows 환경이 맞는지 확인해주세요.")

# --- 2. 데이터 로드 ---
try:
    df = pd.read_csv('gait_full_landmarks.csv')
except FileNotFoundError:
    print("오류: 'gait_full_landmarks.csv' 파일을 찾을 수 없습니다.")
    print("먼저 3-1 단계의 랜드마크 추출 스크립트를 실행해주세요.")
    exit()

# t-SNE 분석에 사용할 랜드마크 좌표 데이터만 선택 (Time_sec 컬럼 제외)
landmark_data = df.drop('Time_sec', axis=1)

if landmark_data.empty:
    print("분석할 데이터가 없습니다.")
    exit()

print("t-SNE 분석을 시작합니다. 데이터 양에 따라 몇 분 정도 소요될 수 있습니다...")

# --- 3. t-SNE 모델 생성 및 학습 ---
# n_components=2 : 2차원으로 축소
# perplexity : 유사 이웃의 수. 보통 5~50 사이의 값을 사용합니다.
# init='pca' : PCA를 사용해 초기화하여 더 안정적인 결과를 얻습니다.
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=42)
pose_embedding = tsne.fit_transform(landmark_data)

print("t-SNE 분석이 완료되었습니다.")

# --- 4. 시각화 ---
plt.figure(figsize=(12, 10))

# c=df['Time_sec'] : 시간에 따라 점의 색깔이 변하도록 설정
# cmap='viridis' : 색상 맵 지정 (노란색에 가까울수록 영상의 뒷부분)
scatter = plt.scatter(
    pose_embedding[:, 0], 
    pose_embedding[:, 1], 
    c=df['Time_sec'], 
    cmap='viridis', 
    alpha=0.6,
    s=10 # 점의 크기
)

# 컬러바 추가
cbar = plt.colorbar(scatter)
cbar.set_label('시간 (초)')

plt.title('t-SNE를 이용한 보행 자세 공간 시각화')
plt.xlabel('t-SNE 차원 1')
plt.ylabel('t-SNE 차원 2')
plt.grid(True)
plt.show()
