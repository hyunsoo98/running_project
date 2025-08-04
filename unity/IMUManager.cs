using UnityEngine;

public class IMUManager : MonoBehaviour
{
    public Animator animator;
    public string imuDeviceName = "Rev_3"; // 사용할 센서 이름
    private Vector3 prevRPY;
    private float updateInterval = 0.1f;
    private float timer = 0f;

    void Start()
    {
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

        Vector3 delta = currentRPY - prevRPY;
        float speed = (Mathf.Abs(delta.x) + Mathf.Abs(delta.y) + Mathf.Abs(delta.z)) / updateInterval;

        // Animator 파라미터에 speed 반영
        animator.SetFloat("Speed", speed);

        prevRPY = currentRPY;
    }
}
