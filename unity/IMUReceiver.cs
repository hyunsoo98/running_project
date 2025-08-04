using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using System.Diagnostics;  // Stopwatch namespace

/// <summary>
/// UDP로 IMU 데이터를 수신해 rotationData에 저장하고, 각 센서별 speed를 계산합니다.
/// </summary>
public class IMUReceiver : MonoBehaviour
{
    private UdpClient client;
    private Thread receiveThread;
    private volatile bool isReceiving;
    private static Stopwatch stopwatch;

    public static Dictionary<string, Vector3> rotationData = new();

    public static float rollRev1, pitchRev1, yawRev1, speedRev1;
    public static float rollRev2, pitchRev2, yawRev2, speedRev2;
    public static float rollRev3, pitchRev3, yawRev3, speedRev3;
    public static float rollRev4, pitchRev4, yawRev4, speedRev4;

    private static Vector3 prevRev1 = Vector3.zero, prevRev2 = Vector3.zero, prevRev3 = Vector3.zero, prevRev4 = Vector3.zero;
    private static float timeRev1 = 0f, timeRev2 = 0f, timeRev3 = 0f, timeRev4 = 0f;

    void Start()
    {
        // UDP 포트 바인딩 및 재사용
        try
        {
            client = new UdpClient();
            client.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            client.Client.Bind(new IPEndPoint(IPAddress.Any, 5005));
        }
        catch (SocketException se)
        {
            UnityEngine.Debug.LogError($"UDP 포트 바인딩 실패: {se.Message}");
            return;
        }

        // 시간 측정용 Stopwatch 시작
        stopwatch = Stopwatch.StartNew();
        isReceiving = true;

        // 백그라운드 스레드에서 수신 시작
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData()
    {
        IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 5005);

        while (isReceiving)
        {
            try
            {
                byte[] data = client.Receive(ref remoteEP);
                string json = Encoding.UTF8.GetString(data);
                IMUData parsed = JsonUtility.FromJson<IMUData>(json);

                if (parsed != null && !string.IsNullOrEmpty(parsed.device))
                {
                    Vector3 current = new Vector3(parsed.roll, parsed.pitch, parsed.yaw);
                    lock (rotationData)
                    {
                        rotationData[parsed.device] = current;
                        UpdateDevice(parsed.device, current);
                    }

                    UnityEngine.Debug.Log($"✅ {parsed.device}: RPY=({parsed.roll}, {parsed.pitch}, {parsed.yaw})");
                }
            }
            catch (SocketException)
            {
                // client.Close() 시 예외 발생 → 루프 종료
                break;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"❌ 데이터 수신/파싱 에러: {e.Message}");
            }
        }
    }

    private void UpdateDevice(string device, Vector3 current)
    {
        float now = (float)stopwatch.Elapsed.TotalSeconds;
        switch (device)
        {
            case "Rev_1": UpdateMetrics(ref prevRev1, ref timeRev1, ref speedRev1, current, now); break;
            case "Rev_2": UpdateMetrics(ref prevRev2, ref timeRev2, ref speedRev2, current, now); break;
            case "Rev_3": UpdateMetrics(ref prevRev3, ref timeRev3, ref speedRev3, current, now); break;
            case "Rev_4": UpdateMetrics(ref prevRev4, ref timeRev4, ref speedRev4, current, now); break;
        }
    }

    private void UpdateMetrics(ref Vector3 prev, ref float time, ref float speed, Vector3 current, float now)
    {
        float dt = now - time;
        if (dt > 0.01f)
        {
            speed = Vector3.Distance(current, prev) / dt;
            prev = current;
            time = now;
        }
    }

    void OnDestroy()
    {
        // 수신 스레드 및 소켓 정리
        isReceiving = false;
        if (client != null) client.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Join();
    }
}