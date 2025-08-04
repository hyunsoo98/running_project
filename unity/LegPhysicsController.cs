// 2025-07-31 AI-Tag
// This was created with the help of Assistant, a Unity Artificial Intelligence product.

using System;
using UnityEditor;
using UnityEngine;

public class LegPhysicsController : MonoBehaviour
{
    public Rigidbody leftLegRigidbody;
    public Rigidbody rightLegRigidbody;
    public Transform leftLegTarget;
    public Transform rightLegTarget;
    public float forceMultiplier = 10f;

    void FixedUpdate()
    {
        // 센서 데이터를 기반으로 다리 움직임 제어
        Vector3 leftForce = (leftLegTarget.position - leftLegRigidbody.position) * forceMultiplier;
        Vector3 rightForce = (rightLegTarget.position - rightLegRigidbody.position) * forceMultiplier;

        leftLegRigidbody.AddForce(leftForce);
        rightLegRigidbody.AddForce(rightForce);
    }
}
