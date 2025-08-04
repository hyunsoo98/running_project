import pandas as pd
from datetime import datetime

# CSV 파일 불러오기
df = pd.read_csv("imu_data1.csv")

# 시간 문자열 → 초 단위로 변환
def time_to_seconds(t):
    try:
        dt = datetime.strptime(t.strip(), "%H:%M:%S.%f")
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
    except Exception as e:
        print(f"시간 변환 오류: '{t}' → {e}")
        return None

# 시간 변환 적용
df["TimeSec"] = df["time"].astype(str).apply(time_to_seconds)

filtered_df = df[(df["TimeSec"] >= start_time_sec) & (df["TimeSec"] < end_time_sec)]

# 결과 저장
filtered_df.to_csv("filtered_17_46_to_48.csv", index=False)

# 미리보기
print(filtered_df.head())
