import asyncio
import tkinter as tk
from bleak import BleakClient, BleakScanner
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time

# 장치 이름 프리픽스를 기반으로 연결
BLE_DEVICES = {
    "Nano_33_Rev_1": "Nano_33_Rev_1",
    "Nano_33_Rev_2": "Nano_33_Rev_2"
}
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1213"

history_len = 100
labels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

root = tk.Tk()
root.title("다중 IMU 실시간 플로팅")
fig, axes = plt.subplots(len(BLE_DEVICES), 3, figsize=(12, 4 * len(BLE_DEVICES)))
fig.tight_layout(pad=3.0)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

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


def make_notification_handler(name):
    def handler(sender, data):
        try:
            decoded = data.decode().strip()
            print(f"[{name} 수신]: {decoded}")
            if "A:" in decoded and "G:" in decoded:
                acc_part, gyr_part = decoded.split("G:")
                ax_values = acc_part.replace("A:", "").strip().split(",")
                gx_values = gyr_part.strip().split(",")

                if len(ax_values) != 3 or len(gx_values) != 3:
                    raise ValueError("누락된 IMU 데이터")

                ax, ay, az = map(float, ax_values)
                gx, gy, gz = map(float, gx_values)

                dd = device_data[name]
                for key, val in zip(labels, [ax, ay, az, gx, gy, gz]):
                    if key in dd and key in device_lines[name]:
                        dd[key].append(val)
                        if len(dd[key]) > history_len:
                            dd[key].pop(0)
                        device_lines[name][key].set_ydata(dd[key])
                canvas.draw_idle()
        except Exception as e:
            print(f"[{name}] ❌ Parse error:", e)
    return handler


async def ble_loop(name, name_prefix):
    print(f"[{name}] 장치 검색 중...")
    found_device = None
    for _ in range(3):
        devices = await BleakScanner.discover(timeout=3.0)
        for d in devices:
            if d.name and name_prefix in d.name:
                found_device = d
                print(f"[{name}] ✅ 발견됨 @ {d.address}")
                break
        if found_device:
            break
        print(f"[{name}] ❌ 재시도 중...")
        await asyncio.sleep(1.0)

    if not found_device:
        print(f"[{name}] ❌ 장치 탐색 실패")
        return

    while True:
        try:
            async with BleakClient(found_device) as client:
                print(f"[{name}] 🔗 연결 시도...")

                # 수동 연결 타임아웃 (10초)
                start_time = time.time()
                while not client.is_connected:
                    if time.time() - start_time > 10:
                        raise Exception("⏰ 연결 타임아웃")
                    await asyncio.sleep(0.5)

                print(f"[{name}] ✅ 연결 성공")

                await client.start_notify(CHARACTERISTIC_UUID, make_notification_handler(name))
                print(f"[{name}] 📡 Notify 시작됨")

                # 연결 유지 루프
                while client.is_connected:
                    await asyncio.sleep(0.1)

                print(f"[{name}] ⚠️ 연결 종료됨, 재시도 예정")

        except Exception as e:
            print(f"[{name}] ❌ 연결 중 예외 발생: {e}")

        # 재연결까지 대기 후 재시도
        await asyncio.sleep(5)


async def main_ble_loop():
    tasks = []
    for name, prefix in BLE_DEVICES.items():
        tasks.append(ble_loop(name, prefix))
        await asyncio.sleep(1.0)  # 연결 간 시간차 적용
    await asyncio.gather(*tasks)


def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_ble_loop())


threading.Thread(target=start_ble_thread, daemon=True).start()
root.mainloop()