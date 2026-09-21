# running_project

IMU/BLE 기반 러닝·워킹 모션 캡처 및 영상 교차검증 프로젝트.
바이오센서(IMU) 데이터의 신뢰성을 영상(MediaPipe) 데이터와 교차검증하고, Unity와 연동해 실시간 모션을 시각화한다.

## 구조

- `ble_test/` — BLE 연결 자체를 검증한 초기 실험 스크립트 (기기별 개별 테스트, 연결 안정성 확인용)
- `ble_iot/` — `ble_test`의 검증을 바탕으로 만든 실제 멀티 디바이스 IMU 수집 파이프라인 (JSON/Unity 연동 수집, 필터 비교·시각화 포함)
- `filtering/` — IMU 노이즈 필터링 (칼만 필터, 상보 필터). 상세 설명은 `filtering/README_IMU_Filtering.md` 참고
- `analysis/` — 모션 분석·시각화 (DTW 유사도 비교, 오일러각, t-SNE, gait landmark, pose space)
- `ml/` — 랜드마크/자세 기반 딥러닝 실험
- `unity/` — IMU 데이터 수신 및 러닝/워킹 모션 애니메이션용 Unity 연동 스크립트 (C#)
- `data/` — 수집된 IMU 로그 및 gait landmark 데이터

## 주요 작업

- 멀티 센서 BLE(Bluetooth Low Energy) 연동 시 발생하는 연결 실패를 방지하기 위한 BLE 수신 루프 최적화
- 센서 데이터 노이즈 제거를 위한 칼만 필터·상보 필터 적용 및 캘리브레이션
- IMU 데이터와 영상(MediaPipe 하반신 랜드마크) 간 DTW 기반 시계열 교차검증
- Unity 엔진과 센서 데이터 연동을 통한 실시간 러닝/워킹 모션 시각화 (speed 값 기반 모션 변동 도출)

## 사용 센서

Arduino Nano 33 BLE Sense Rev2 x2, Arduino Nano 33 IoT x2

## 관련 프로젝트

Unity 프로젝트 코드: [run_unity](https://github.com/hyunsoo98/run_unity)
