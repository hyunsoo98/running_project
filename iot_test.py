import asyncio
import tkinter as tk
from bleak import BleakClient
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading

# 여러 Nano 33 IoT 장치 주소
BLE_DEVICES = {
    "Nano_A": "58:BF:25:9B:F2:02",
    "Nano_B": "EC:62:60:82:DB:DA"
}
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1213"

history_len = 100
labels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

# tkinter 설정
root = tk.Tk()
root.title("다중 IMU 실시간 플로팅")
fig, axes = plt.subplots(len(BLE_DEVICES), 3, figsize=(12, 4 * len(BLE_DEVICES)))
fig.tight_layout(pad=3.0)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 장치별 데이터 관리
device_data = {}
device_lines = {}

for i, (name, _) in enumerate(BLE_DEVICES.items()):
    imu_vals = {key: [0]*history_len for key in labels}
    device_data[name] = imu_vals

    subplot_axes = axes[i] if len(BLE_DEVICES) > 1 else [axes]
    lines = {}
    for ax, label in zip(subplot_axes, labels):
        ax.set_title(f"{name} - {label}")
        ax.set_xlim(0, history_len)
        ax.set_ylim(-250, 250) if label.startswith("G") else ax.set_ylim(-2, 2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        lines[label], = ax.plot(range(history_len), imu_vals[label])
    device_lines[name] = lines

# Notify 핸들러
def make_notification_handler(name):
    def handler(sender, data):
        try:
            decoded = data.decode().strip()
            print(f"[{name} 수신]: {decoded}")
            if "A:" in decoded and "G:" in decoded:
                acc_part, gyr_part = decoded.split("G:")
                ax, ay, az = map(float, acc_part.replace("A:", "").strip().split(","))
                gx, gy, gz = map(float, gyr_part.strip().split(","))

                dd = device_data[name]
                for key, val in zip(labels, [ax, ay, az, gx, gy, gz]):
                    dd[key].append(val)
                    if len(dd[key]) > history_len:
                        dd[key].pop(0)
                    device_lines[name][key].set_ydata(dd[key])
                canvas.draw_idle()
        except Exception as e:
            print(f"[{name}] ❌ Parse error:", e)
    return handler

# 각 BLE 장치 접속 루프
async def ble_loop(name, address):
    async with BleakClient(address) as client:
        print(f"[{name}] BLE 연결 시도 중...")
        await client.connect()
        if not client.is_connected:
            print(f"[{name}] ❌ 연결 실패")
            return

        print(f"[{name}] ✅ 연결됨")
        await client.start_notify(CHARACTERISTIC_UUID, make_notification_handler(name))
        while True:
            await asyncio.sleep(0.1)

# 모든 BLE 루프 실행
async def main_ble_loop():
    tasks = [ble_loop(name, addr) for name, addr in BLE_DEVICES.items()]
    await asyncio.gather(*tasks)

def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_ble_loop())

threading.Thread(target=start_ble_thread, daemon=True).start()
root.mainloop()
