using UnityEngine;

/// <summary>
/// IMU 회전값 기반 속도를 빠르게 계산하고 즉시 반영하여,
/// 걷기/뛰기 애니메이션 전환에 딜레이가 없도록 구성합니다.
/// </summary>
public class IMUSpeedAnimator : MonoBehaviour
{
    [Header("Animator (동일 GameObject)")]
    public Animator animator;

    [Header("IMU 센서 이름 (Rev_1 ~ Rev_4)")]
    public string imuDeviceName = "Rev_3";

    [Header("다리 길이 (m, 센서 위치 ~ 관절)")]
    public float limbLength = 0.9f;

    [Header("업데이트 간격 (초)")]
    public float updateInterval = 0.05f;

    [Header("속도 반영 반응 시간 (감속 포함)")]
    public float smoothTime = 0.05f;  // 감속도 빠르게 반응하도록 낮게 설정

    private Vector3 prevRPY;
    private float timer = 0f;
    private float displayedSpeed = 0f;
    private float smoothVel = 0f;

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();

        if (IMUReceiver.rotationData.ContainsKey(imuDeviceName))
            prevRPY = IMUReceiver.rotationData[imuDeviceName];
        else
            prevRPY = Vector3.zero;
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer < updateInterval) return;
        timer = 0f;

        Vector3 currentRPY;
        lock (IMUReceiver.rotationData)
        {
            if (!IMUReceiver.rotationData.ContainsKey(imuDeviceName)) return;
            currentRPY = IMUReceiver.rotationData[imuDeviceName];
        }

        float dx = Mathf.DeltaAngle(prevRPY.x, currentRPY.x);
        float dy = Mathf.DeltaAngle(prevRPY.y, currentRPY.y);
        float dz = Mathf.DeltaAngle(prevRPY.z, currentRPY.z);

        Vector3 delta = new Vector3(dx, dy, dz);
        float degPerSec = delta.magnitude / updateInterval;
        float omegaRad = degPerSec * Mathf.Deg2Rad;
        float linearSpeed = omegaRad * limbLength;
        float targetSpeed = linearSpeed * 3.6f; // m/s → km/h

        // 가감속 모두 빠르게 반응
        displayedSpeed = Mathf.SmoothDamp(displayedSpeed, targetSpeed, ref smoothVel, smoothTime);

        animator.SetFloat("Speed", displayedSpeed);
        prevRPY = currentRPY;
    }
}
