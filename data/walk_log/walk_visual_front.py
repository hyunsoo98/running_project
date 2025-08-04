import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 하체 관절 연결 리스트
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

# --- 실시간 웹캠 처리 시작 ---
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# novità: 영상 저장을 위한 설정
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# 웹캠의 FPS를 정확히 가져오지 못하는 경우가 있어, 20으로 고정. 필요시 조절 가능
fps = 30.0 
out = cv2.VideoWriter('output_live.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

analysis_data = []
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
        h, w, _ = image.shape
        
        # (이전과 동일한 관절 좌표 추출 및 계산 로직)
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        step_width_pixel = abs(left_ankle[0] - right_ankle[0]) * w
        
        elapsed_time = time.time() - start_time
        analysis_data.append([elapsed_time, left_knee_angle, right_knee_angle, step_width_pixel])
        
        cv2.putText(image, f"L.Knee: {int(left_knee_angle)}", (int(left_knee[0]*w) - 150, int(left_knee[1]*h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"R.Knee: {int(right_knee_angle)}", (int(right_knee[0]*w) + 20, int(right_knee[1]*h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Step Width: {int(step_width_pixel)} px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except:
        pass

    mp_drawing.draw_landmarks(image, results.pose_landmarks, LOWER_BODY_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
    
    # novità: 처리된 프레임을 동영상 파일에 쓰기
    out.write(image)
    
    cv2.imshow('Real-time Analysis (Recording...)', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 작업 완료 후 리소스 해제
cap.release()
out.release() # novità: 비디오 저장 객체 해제
cv2.destroyAllWindows()

# 수집된 데이터를 CSV 파일로 저장
if analysis_data:
    df_analysis = pd.DataFrame(analysis_data, columns=['Time_sec', 'Left_Knee_Angle', 'Right_Knee_Angle', 'Step_Width_px'])
    df_analysis.to_csv('realtime_gait_analysis.csv', index=False)
    print("\n✅ 분석 종료! 'realtime_gait_analysis.csv'와 'output_live.mp4' 파일이 저장되었습니다.")