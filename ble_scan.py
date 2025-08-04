import asyncio
from bleak import BleakScanner, BleakClient

TARGET_MAC = "10:32:93:5B:FC:D9"  # NanoBLE_Test 주소

async def scan_and_connect():
    print("🔍 BLE 디바이스 스캔 중...")
    devices = await BleakScanner.discover(timeout=5.0)

    # MAC 주소로 대상 장치 직접 탐색
    target = next((d for d in devices if d.address.upper() == TARGET_MAC), None)

    if not target:
        print("❌ NanoBLE_Test를 발견하지 못했습니다.")
        return

    print(f"✅ NanoBLE_Test 발견됨: {target.address}, 연결 시도 중...")

    try:
        async with BleakClient(target.address) as client:
            if not client.is_connected:
                print("❌ 연결 실패")
                return

            print("✅ 연결 성공")
            services = await client.get_services()
            for s in services:
                print(f"- 서비스: {s.uuid}")
    except Exception as e:
        print(f"❌ 예외 발생: {e}")

asyncio.run(scan_and_connect())
