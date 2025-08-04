import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# --- 1. 설정 및 초기화 ---

# 분석할 영상 파일 경로를 지정해주세요.
VIDEO_FILE_PATH = 'output_live.mp4' 

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 하체 관절 연결 리스트
LOWER_BODY_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- 2. 영상 파일 처리 ---
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_FILE_PATH}")
    exit()

# 영상의 FPS와 프레임 크기 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 분석 결과를 저장할 동영상 파일 설정
# out = cv2.VideoWriter('output_video_analysis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

analysis_data = []
frame_count = 0

print(f"Starting analysis on {VIDEO_FILE_PATH} ({total_frames} frames, {fps:.2f} FPS)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing all frames.")
        break

    # 프레임 번호 및 시간 계산
    frame_count += 1
    elapsed_time = frame_count / fps

    # MediaPipe 처리를 위해 이미지 색상 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        
        # 관절 좌표 추출
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # 각도 및 보폭 계산
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        step_width_pixel = abs(left_ankle[0] - right_ankle[0]) * frame_width
        
        # 분석 데이터에 발목의 y좌표 추가
        left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        
        analysis_data.append([elapsed_time, left_knee_angle, right_knee_angle, step_width_pixel, left_ankle_y, right_ankle_y])
        
    except Exception as e:
        # 랜드마크 감지에 실패한 경우, 데이터를 추가하지 않고 넘어감
        # print(f"Frame {frame_count}: Could not detect landmarks. Skipping.")
        pass

    # 진행 상황 출력 (100 프레임마다)
    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}/{total_frames}...")

    # (선택 사항) 처리 과정을 화면에 보여주려면 아래 주석을 해제하세요.
    # 단, 처리 속도가 매우 느려집니다.
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, LOWER_BODY_CONNECTIONS)
    # cv2.imshow('Video Analysis', image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# --- 3. 결과 저장 ---
cap.release()
# out.release()
cv2.destroyAllWindows()

# 수집된 데이터를 CSV 파일로 저장
if analysis_data:
    df_analysis = pd.DataFrame(analysis_data, columns=['Time_sec', 'Left_Knee_Angle', 'Right_Knee_Angle', 'Step_Width_px', 'Left_Ankle_y', 'Right_Ankle_y'])
    df_analysis.to_csv('gait_analysis_with_coords.csv', index=False)
    print("\n✅ Analysis finished! 'gait_analysis_with_coords.csv' has been saved.")
else:
    print("\n⚠️ No data was collected. Check if landmarks were detected in the video.")

