import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# novità: 하체 관절만 연결하기 위한 커스텀 연결 리스트 생성
LOWER_BODY_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
    (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
    (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
]


def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calculate_pitch_angle(knee, ankle):
    knee = np.array(knee); ankle = np.array(ankle)
    vertical_point = (knee[0], knee[1] - 100)
    return calculate_angle(vertical_point, knee, ankle)


cap = cv2.VideoCapture(1) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

angle_data = []
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        shin_pitch = calculate_pitch_angle(left_knee, left_ankle)
        elapsed_time = time.time() - start_time
        angle_data.append([elapsed_time, shin_pitch])
        
        h, w, _ = image.shape
        cv2.putText(image, f"Shin Pitch: {int(shin_pitch)}", 
                    (int(left_knee[0]*w) + 10, int(left_knee[1]*h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        pass

    # novità: 기본 POSE_CONNECTIONS 대신 우리가 만든 LOWER_BODY_CONNECTIONS 사용
    mp_drawing.draw_landmarks(image, results.pose_landmarks, LOWER_BODY_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
    
    cv2.imshow('Lower Body Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

if angle_data:
    df_angles = pd.DataFrame(angle_data, columns=['Time_sec', 'Shin_Pitch_Angle'])
    df_angles.to_csv('realtime_lowerbody_angles.csv', index=False)
    print("\n✅ 분석 종료! 'realtime_lowerbody_angles.csv' 파일이 저장되었습니다.")