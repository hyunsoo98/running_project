// 2025-07-31 AI-Tag
// This was created with the help of Assistant, a Unity Artificial Intelligence product.
// Refactored for stability and correctness.

using UnityEngine;

public class GenericIMUController : MonoBehaviour
{
    [Header("센서 이름 (Rev_1 ~ Rev_4)")]
    public string deviceName = "Rev_3";

    [Header("Pitch 제한 (무릎 굽힘 각도, deg)")]
    public float minPitch = -10f;
    public float maxPitch = 60f;

    [Header("초기 오프셋 (Euler)")]
    public Vector3 initialOffset = Vector3.zero;

    [Header("회전 변화 최대 속도 (deg/frame)")]
    public float maxRotationPerFrame = 20f;

    [Header("Animator (동일 GameObject)")]
    public Animator animator;

    [Header("다리 길이 (m, 센서 위치 ~ 관절)")]
    public float limbLength = 0.9f;

    [Header("업데이트 간격 (초)")]
    public float updateInterval = 0.05f;

    [Header("속도 반영 반응 시간 (감속 포함)")]
    public float smoothTime = 0.05f;

    private Quaternion sensorInitialRotation = Quaternion.identity; // [개선됨] 기준 회전값
    private Quaternion offsetQuat;
    private bool isCalibrated = false;

    private Vector3 prevRPY;
    private float timer = 0f;
    private float displayedSpeed = 0f;
    private float smoothVel = 0f;

    // [수정됨] Pitch 데이터를 일관성 있게 rpy.x에서 가져옴
    public float CurrentPitch
    {
        get
        {
            if (IMUReceiver.rotationData.ContainsKey(deviceName))
            {
                lock (IMUReceiver.rotationData)
                {
                    // HandleRotation과 동일하게 rpy.x를 Pitch로 사용
                    return IMUReceiver.rotationData[deviceName].x;
                }
            }
            return 0f;
        }
    }

    void Awake()
    {
        // [개선됨] GetComponent는 Start보다 Awake에서 하는 것이 안전합니다.
        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }
        offsetQuat = Quaternion.Euler(initialOffset);
    }
    
    // [개선됨] 사용자가 원할 때 초기화(Calibration)를 실행하도록 변경
    void Update()
    {
        // 'C' 키를 눌러서 현재 센서 자세를 기준으로 보정
        if (Input.GetKeyDown(KeyCode.C))
        {
            Calibrate();
        }
    }

    void LateUpdate()
    {
        if (!isCalibrated || !IMUReceiver.rotationData.ContainsKey(deviceName)) return;

        Vector3 rpy;
        lock (IMUReceiver.rotationData)
        {
            rpy = IMUReceiver.rotationData[deviceName];
        }

        // [수정됨] 회전 적용 로직을 별도 함수에서 호출
        ApplySensorRotation(rpy);

        // 속도 계산 및 애니메이션 제어
        HandleSpeed(rpy);
    }

    /// <summary>
    /// 현재 센서의 회전을 기준으로 영점을 설정합니다.
    /// </summary>
    public void Calibrate()
    {
        if (!IMUReceiver.rotationData.ContainsKey(deviceName))
        {
            Debug.LogWarning($"'{deviceName}' 센서 데이터를 찾을 수 없어 초기화에 실패했습니다.");
            return;
        }

        lock (IMUReceiver.rotationData)
        {
            Vector3 rpy = IMUReceiver.rotationData[deviceName];
            // Pitch 값만 사용하여 초기 회전 기준 설정
            sensorInitialRotation = Quaternion.Euler(rpy.x, 0, 0); 
        }

        isCalibrated = true;
        Debug.Log($"'{deviceName}' 센서 초기화 완료!");
    }

    private void ApplySensorRotation(Vector3 rpy)
    {
        // [수정됨] 기존의 불안정한 회전 누적 방식 대신, 애니메이션 위에 덮어쓰는 방식으로 변경
        
        // 1. 현재 프레임의 애니메이션에 의해 계산된 기본 회전 값을 저장
        Quaternion baseAnimationRotation = transform.localRotation;

        // 2. 센서 값으로 목표 회전(상대값) 계산
        float clampedPitch = Mathf.Clamp(rpy.x, minPitch, maxPitch);
        Quaternion currentSensorRotation = Quaternion.Euler(clampedPitch, 0, 0);

        // 초기 기준 회전으로부터의 상대적인 변화량 계산
        Quaternion relativeRotation = Quaternion.Inverse(sensorInitialRotation) * currentSensorRotation;

        // 최종 목표 회전 = 오프셋 * 센서 상대 회전
        Quaternion targetRotation = offsetQuat * relativeRotation;

        // 3. 기본 애니메이션 회전 값에 센서 회전을 곱하여 최종 회전 결정
        transform.localRotation = baseAnimationRotation * targetRotation;
    }

    private void HandleSpeed(Vector3 currentRPY)
    {
        timer += Time.deltaTime;
        if (timer < updateInterval) return;

        float angularSpeed = Vector3.Angle(prevRPY, currentRPY) / timer; // deg/s
        float omegaRad = angularSpeed * Mathf.Deg2Rad; // rad/s
        float linearSpeed = omegaRad * limbLength; // m/s
        float targetSpeedKmh = linearSpeed * 3.6f; // km/h

        displayedSpeed = Mathf.SmoothDamp(displayedSpeed, targetSpeedKmh, ref smoothVel, smoothTime);

        animator.SetFloat("Speed", displayedSpeed);

        // 리셋
        prevRPY = currentRPY;
        timer = 0f;
    }
}