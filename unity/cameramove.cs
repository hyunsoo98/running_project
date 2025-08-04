using UnityEngine;

/// <summary>
/// 3인칭 시점으로 target을 위 대각선 뒤에서 부드럽게 따라가며 바라보는 카메라.
/// </summary>
public class ThirdPersonCameraController : MonoBehaviour
{
    [Header("따라갈 대상")]
    public Transform target;              // 주인공 Transform

    [Header("카메라 오프셋")]
    public Vector3 offset = new Vector3(0f, 3f, -5f);
    [Header("부드러움 (0~1)")]
    [Range(0.01f,1f)]
    public float smoothSpeed = 0.125f;    // 클수록 천천히 따라옴

    void LateUpdate()
    {
        if (target == null) return;

        // 1) 목표 위치 + 오프셋
        Vector3 desiredPosition = target.position + offset;
        // 2) 부드럽게 보간
        Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed);
        transform.position = smoothedPosition;

        // 3) 항상 target을 바라보게
        Vector3 lookAtPoint = target.position + Vector3.up * (offset.y * 0.3f);
        transform.LookAt(lookAtPoint);
    }
}
