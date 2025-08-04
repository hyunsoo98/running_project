import re
import csv

# 파일 경로
input_file = "C:/Users/asia/Desktop/vscode/pigeon_project/walk_log/Log 2025-07-22 16_31_44.txt"
output_file = "C:/Users/asia/Desktop/vscode/pigeon_project/walk_log/imu_data6.csv"

# 정규식: 시간, R, P, Y 추출
pattern = re.compile(r'^A\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+".*?R:([-\d.]*),P:([-\d.]*),Y:([-\d.]*)"')

imu_data = []

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        match = pattern.match(line)
        if match:
            time_str, roll, pitch, yaw = match.groups()

            # 각 값에 대해 null 처리
            def parse_or_null(val):
                try:
                    return float(val)
                except:
                    return None

            roll_f = parse_or_null(roll)
            pitch_f = parse_or_null(pitch)
            yaw_f = parse_or_null(yaw)

            imu_data.append([time_str, roll_f, pitch_f, yaw_f])

# CSV로 저장
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time", "Roll", "Pitch", "Yaw"])
    writer.writerows(imu_data)

print(f"✅ CSV 저장 완료: {output_file} (총 {len(imu_data)}개)")
