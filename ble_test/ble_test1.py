import asyncio
import tkinter as tk
from bleak import BleakClient
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
ADDRESS = "64:5B:B3:8B:A5:08"
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1213"
# 데이터 저장
history_len = 100
x_vals = list(range(history_len))
imu_data = {
    'Ax': [0]*history_len, 'Ay': [0]*history_len, 'Az': [0]*history_len,
    'Gx': [0]*history_len, 'Gy': [0]*history_len, 'Gz': [0]*history_len
}
# tkinter GUI 설정
root = tk.Tk()
root.title("IMU 6축 실시간 플로팅")
root.geometry("900x600")
fig, axes = plt.subplots(2, 3, figsize=(10, 5))
fig.tight_layout(pad=3.0)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
# subplot 연결
lines = {}
labels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
for ax, label in zip(axes.flat, labels):
    ax.set_title(label)
    ax.set_xlim(0, history_len)
    # :흰색_확인_표시: 자이로 범위 확대: ±500 dps 또는 ±250도 설정 가능
    if label.startswith("G"):
        ax.set_ylim(-250, 250)  # 자이로 범위 확대
    else:
        ax.set_ylim(-2, 2)      # 가속도는 그대로
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    lines[label], = ax.plot(x_vals, imu_data[label])
# BLE Notify 핸들러
def handle_notification(sender, data):
    try:
        decoded = data.decode()
        print(f":글씨가_쓰여진_페이지: Decoded: {decoded}")
        if decoded.startswith("A:"):
            acc_part, gyr_part = decoded.split("G:")
            ax, ay, az = map(float, acc_part.replace("A:", "").split(","))
            gx, gy, gz = map(float, gyr_part.strip().split(","))
            imu_data['Ax'].append(ax)
            imu_data['Ay'].append(ay)
            imu_data['Az'].append(az)
            imu_data['Gx'].append(gx)
            imu_data['Gy'].append(gy)
            imu_data['Gz'].append(gz)
            # 슬라이딩 윈도우 유지
            for key in imu_data:
                if len(imu_data[key]) > history_len:
                    imu_data[key].pop(0)
                lines[key].set_ydata(imu_data[key])
            canvas.draw()
    except Exception as e:
        print(":x: Parse error:", e)
# BLE 루프
async def ble_loop():
    async with BleakClient(ADDRESS) as client:
        print(":링크: BLE 연결 중...")
        await client.start_notify(CHARACTERISTIC_UUID, handle_notification)
        print(":흰색_확인_표시: Notify 수신 중...")
        while True:
            await asyncio.sleep(0.1)
# tkinter + asyncio 통합
def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_loop())
threading.Thread(target=start_ble_thread, daemon=True).start()
root.mainloop()