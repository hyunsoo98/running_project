import asyncio
from bleak import BleakScanner, BleakClient

async def explore_ble():
    print("🔍 주변 BLE 기기 검색 중 (5초)...")
    devices = await BleakScanner.discover(timeout=5.0)

    if not devices:
        print("❌ BLE 장치가 감지되지 않았습니다.")
        return

    print("\n=== 검색된 BLE 장치 목록 ===")
    for i, device in enumerate(devices):
        print(f"[{i}] {device.name or '이름 없음'} ({device.address})")

    idx = int(input("\n📌 연결할 장치 번호를 선택하세요: "))
    selected = devices[idx]
    print(f"\n🔗 '{selected.name}' ({selected.address})에 연결 중...\n")

    async with BleakClient(selected.address) as client:
        services = await client.get_services()
        print("📋 사용 가능한 서비스 및 특성:")
        for service in services:
            print(f"[Service] {service.uuid}")
            for char in service.characteristics:
                props = ",".join(char.properties)
                print(f" └─ [Characteristic] {char.uuid} | 속성: {props}")

asyncio.run(explore_ble())