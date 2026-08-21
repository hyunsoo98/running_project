# running_project

IMU/BLE 기반 러닝 모션 캡처·분석 프로젝트.

## 구조

- `filtering/` — IMU 노이즈 필터링(자세한 설명은 `filtering/README_IMU_Filtering.md` 참고)
- `analysis/` — 모션 분석/시각화(DTW, 오일러각, t-SNE, gait landmark, pose space)
- `ml/` — 랜드마크/자세 기반 딥러닝 실험
- `ble_iot/` — BLE 멀티디바이스 IMU 수집 파이프라인 + 필터링 비교/시각화
- `ble_test/` — BLE 연결 자체를 검증하던 초기 실험 스크립트(장비별 개별 테스트)
- `data/` — 수집된 IMU 로그
- `unity/` — Unity 연동 쪽 리소스

`ble_iot`와 `ble_test`가 둘 다 있는 이유: `ble_test`는 기기 연결이 되는지 확인하던
초기 실험이고, `ble_iot`는 그걸 바탕으로 만든 실제 멀티디바이스 수집 파이프라인으로
보입니다 — 코드만으로 확실하진 않아 일단 둘 다 남겨뒀습니다. 정리하려면 어느 쪽이
지금 실제로 쓰는 코드인지 알려주세요.
