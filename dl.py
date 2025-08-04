import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# --- 1. 한글 폰트 설정 ---
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Malgun Gothic 폰트를 찾을 수 없습니다. Windows 환경이 맞는지 확인해주세요.")

# --- 2. 데이터 로드 및 전처리 ---
try:
    df_train = pd.read_csv('imu_data3.csv')
    df_test = pd.read_csv('imu_data5.csv')
except FileNotFoundError as e:
    print(f"오류: {e.filename} 파일을 찾을 수 없습니다.")
    exit()

# [수정된 부분] 데이터에 포함된 결측치(NaN)를 0으로 대체합니다.
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

# 분석에 사용할 컬럼 (Roll, Pitch, Yaw)
features = ['Roll', 'Pitch', 'Yaw']
train_data = df_train[features]
test_data = df_test[features]

# 데이터 정규화 (0~1 사이의 값으로 변환)
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

print(f"훈련 데이터(기준 보행) 형태: {scaled_train_data.shape}")
print(f"테스트 데이터(비교 보행) 형태: {scaled_test_data.shape}")

# --- 3. 오토인코더 모델 정의 및 훈련 ---
input_dim = len(features)

input_layer = Input(shape=(input_dim,))
encoder = Dense(2, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(input_layer)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mae')

print("\n'기준 보행(imu_data3)'으로 오토인코더 모델 훈련을 시작합니다...")
autoencoder.fit(scaled_train_data, scaled_train_data,
                epochs=50,
                batch_size=32,
                shuffle=True,
                verbose=0)
print("모델 훈련 완료.")

# --- 4. 복원 오차 계산 ---
train_reconstruction = autoencoder.predict(scaled_train_data)
train_error = np.mean(np.abs(scaled_train_data - train_reconstruction), axis=1)

test_reconstruction = autoencoder.predict(scaled_test_data)
test_error = np.mean(np.abs(scaled_test_data - test_reconstruction), axis=1)

print("\n복원 오차 계산 완료.")
print(f"기준 보행 평균 오차: {np.mean(train_error):.4f}")
print(f"비교 보행 평균 오차: {np.mean(test_error):.4f}")

# --- 5. 결과 시각화 ---
plt.figure(figsize=(12, 7))

plt.hist(train_error, bins=50, density=True, alpha=0.7, label='기준 보행 오차 (imu_data3)')
plt.hist(test_error, bins=50, density=True, alpha=0.7, label='비교 보행 오차 (imu_data5)')

plt.title('딥러닝 모델의 보행 상태 분별력 검증', fontsize=18)
plt.xlabel('복원 오차 (값이 클수록 모델에게 "낯선" 데이터)', fontsize=12)
plt.ylabel('데이터 분포 밀도', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
