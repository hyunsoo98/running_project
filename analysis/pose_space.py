import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# --- 1. 설정 ---
VIDEO_FILE_PATH = 'output_live.mp4' # 분석할 영상 파일 경로를 지정해주세요.
OUTPUT_CSV_PATH = 'gait_full_landmarks.csv' # 저장될 CSV 파일 이름

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 분석에 사용할 하체 랜드마크 리스트
LOWER_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

# --- 2. 영상 파일 처리 ---
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    print(f"오류: 영상 파일을 열 수 없습니다: {VIDEO_FILE_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_landmarks_data = []
frame_count = 0

print(f"{VIDEO_FILE_PATH} 영상 분석을 시작합니다. (총 {total_frames} 프레임, {fps:.2f} FPS)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = frame_count / fps

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True

    try:
        landmarks = results.pose_landmarks.landmark
        
        # 현재 프레임의 모든 랜드마크 좌표를 저장할 리스트
        current_frame_landmarks = [elapsed_time]
        
        for landmark_enum in LOWER_BODY_LANDMARKS:
            # 각 랜드마크의 이름과 x, y 좌표를 리스트에 추가
            # 예: 'LEFT_HIP_x', 0.54, 'LEFT_HIP_y', 0.87
            current_frame_landmarks.append(landmarks[landmark_enum.value].x)
            current_frame_landmarks.append(landmarks[landmark_enum.value].y)
            
        all_landmarks_data.append(current_frame_landmarks)

    except Exception as e:
        # 랜드마크 감지 실패 시 건너뛰기
        pass

    if frame_count % 100 == 0:
        print(f"프레임 처리 중: {frame_count}/{total_frames}...")

# --- 3. 결과 저장 ---
cap.release()
print("영상 처리가 완료되었습니다.")

if all_landmarks_data:
    # CSV 파일 헤더(컬럼명) 생성
    columns = ['Time_sec']
    for landmark_enum in LOWER_BODY_LANDMARKS:
        columns.append(f'{landmark_enum.name}_x')
        columns.append(f'{landmark_enum.name}_y')
        
    df_landmarks = pd.DataFrame(all_landmarks_data, columns=columns)
    df_landmarks.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ 분석 완료! '{OUTPUT_CSV_PATH}' 파일이 저장되었습니다.")
else:
    print("\n⚠️ 데이터를 수집하지 못했습니다. 영상에서 랜드마크가 감지되었는지 확인해주세요.")

