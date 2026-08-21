import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# --- 1. 한글 폰트 및 환경 설정 ---
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Malgun Gothic 폰트를 찾을 수 없습니다. Windows 환경이 맞는지 확인해주세요.")

# --- 2. 데이터 로드 및 전처리 ---
try:
    df = pd.read_csv('gait_full_landmarks.csv')
except FileNotFoundError:
    print("오류: 'gait_full_landmarks.csv' 파일을 찾을 수 없습니다.")
    print("먼저 3-1 단계의 랜드마크 추출 스크립트를 실행해주세요.")
    exit()

# Time_sec 컬럼을 제외한 랜드마크 데이터 사용
landmark_data = df.drop('Time_sec', axis=1)

# 데이터 정규화 (0~1 사이의 값으로 변환)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(landmark_data)

print(f"데이터 준비 완료. 데이터 형태: {scaled_data.shape}")

# --- 3. 오토인코더 모델 정의 ---
# 입력 데이터의 차원 (랜드마크 개수 * 2)
input_dim = scaled_data.shape[1]
encoding_dim = 8  # 데이터를 압축할 차원 크기

# 입력 레이어
input_layer = Input(shape=(input_dim,))

# 인코더 부분 (점점 차원을 줄여나감)
encoder = Dense(16, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder) # 압축된 표현 (Bottleneck)

# 디코더 부분 (다시 원래 차원으로 복원)
decoder = Dense(16, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder) # sigmoid는 0~1 사이 값을 출력

# 오토인코더 모델 생성
autoencoder = Model(inputs=input_layer, outputs=decoder)

# 모델 컴파일
autoencoder.compile(optimizer='adam', loss='mae') # Mean Absolute Error
autoencoder.summary()

# --- 4. 모델 훈련 ---
# 중요: 제공된 영상의 데이터가 '정상'이라고 가정하고 전체 데이터로 학습합니다.
print("\n오토인코더 모델 훈련을 시작합니다...")
history = autoencoder.fit(scaled_data, scaled_data,
                          epochs=50,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.1, # 10%는 검증용으로 사용
                          verbose=1)
print("모델 훈련 완료.")

# --- 5. 이상 탐지 (복원 오차 계산) ---
# 훈련된 모델로 데이터를 예측(복원)
reconstructed_data = autoencoder.predict(scaled_data)

# 원본과 복원된 데이터 간의 오차(MAE) 계산
reconstruction_error = np.mean(np.abs(scaled_data - reconstructed_data), axis=1)

# 오차 데이터프레임 생성
error_df = pd.DataFrame({'Time_sec': df['Time_sec'], 'error': reconstruction_error})

# 이상치(Anomaly) 기준점(Threshold) 설정
# 여기서는 평균 + (2 * 표준편차)를 기준점으로 사용. 이 값은 조절 가능합니다.
threshold = np.mean(error_df['error']) + 2 * np.std(error_df['error'])
anomalies = error_df[error_df['error'] > threshold]

print(f"\n이상 탐지 기준(Threshold): {threshold:.4f}")
print(f"총 {len(anomalies)}개의 이상 프레임이 감지되었습니다.")

# --- 6. 결과 시각화 ---
# 1. 시간에 따른 복원 오차 그래프
plt.figure(figsize=(15, 6))
plt.plot(error_df['Time_sec'], error_df['error'], label='복원 오차')
plt.axhline(y=threshold, color='r', linestyle='--', label='이상 탐지 기준점')
plt.scatter(anomalies['Time_sec'], anomalies['error'], color='red', label='이상치')
plt.title('시간에 따른 복원 오차 및 이상 탐지')
plt.xlabel('시간 (초)')
plt.ylabel('복원 오차')
plt.legend()
plt.show()

# 2. t-SNE 공간에 이상치 표시
print("\nt-SNE 시각화를 다시 생성합니다 (이상치 표시)...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=42)
pose_embedding = tsne.fit_transform(landmark_data)

# 정상/이상 여부를 나타내는 라벨 생성
is_anomaly = error_df['error'] > threshold

plt.figure(figsize=(12, 10))
# 정상 데이터 플로팅
plt.scatter(pose_embedding[~is_anomaly, 0], pose_embedding[~is_anomaly, 1], 
            c='blue', alpha=0.4, s=10, label='정상')
# 이상 데이터 플로팅
plt.scatter(pose_embedding[is_anomaly, 0], pose_embedding[is_anomaly, 1], 
            c='red', alpha=1.0, s=20, label='이상')

plt.title('t-SNE 공간에서의 이상 탐지 결과 시각화')
plt.xlabel('t-SNE 차원 1')
plt.ylabel('t-SNE 차원 2')
plt.legend()
plt.grid(True)
plt.show()

