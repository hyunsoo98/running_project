import asyncio
import tkinter as tk
from bleak import BleakClient
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading

ADDRESS = "58:BF:25:9B:F2:02"
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
    ax.set_ylim(-250, 250) if label.startswith("G") else ax.set_ylim(-2, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    lines[label], = ax.plot(x_vals, imu_data[label])

# BLE Notify 핸들러
def handle_notification(sender, data):
    try:
        decoded = data.decode().strip()
        print(f"[BLE 수신]: {decoded}")

        if "A:" in decoded and "G:" in decoded:
            acc_part, gyr_part = decoded.split("G:")
            ax, ay, az = map(float, acc_part.replace("A:", "").strip().split(","))
            gx, gy, gz = map(float, gyr_part.strip().split(","))

            imu_data['Ax'].append(ax)
            imu_data['Ay'].append(ay)
            imu_data['Az'].append(az)
            imu_data['Gx'].append(gx)
            imu_data['Gy'].append(gy)
            imu_data['Gz'].append(gz)

            for key in imu_data:
                if len(imu_data[key]) > history_len:
                    imu_data[key].pop(0)
                lines[key].set_ydata(imu_data[key])
            canvas.draw_idle()
    except Exception as e:
        print("❌ Parse error:", e)

# BLE 루프
async def ble_loop():
    async with BleakClient(ADDRESS) as client:
        print(":링크: BLE 연결 시도 중...")
        await client.connect()
        await asyncio.sleep(1.0)

        if not client.is_connected:
            print("❌ 연결 실패")
            return

        print("✅ 연결 성공! 서비스 로딩 중...")
        await asyncio.sleep(1.0)

        try:
            services = await client.get_services()
            print("📋 GATT 서비스 준비 완료")
        except Exception as e:
            print(f"❌ 서비스 로딩 실패: {e}")
            return

        await asyncio.sleep(1.0)  # Notify 직전 안정화 대기

        for i in range(3):
            try:
                await client.start_notify(CHARACTERISTIC_UUID, handle_notification)
                print("📡 Notify 시작됨")
                break
            except Exception as e:
                print(f"⚠️ Notify 실패 재시도 {i+1}/3: {e}")
                await asyncio.sleep(1.5)
        else:
            print("❌ Notify 시작 실패")
            return

        while True:
            await asyncio.sleep(0.1)

# tkinter + asyncio 통합
def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_loop())

threading.Thread(target=start_ble_thread, daemon=True).start()
root.mainloop()
