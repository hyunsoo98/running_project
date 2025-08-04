using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems; // UI 시스템 사용을 위해 추가

// 이 스크립트는 CharacterController와 Animator가 반드시 필요하다고 명시합니다.
[RequireComponent(typeof(CharacterController))]
[RequireComponent(typeof(Animator))]
public class PlayerController : MonoBehaviour
{
    [Header("수동 연결 컴포넌트")]
    public GenericIMUController kneeController; // 무릎 센서는 다른 오브젝트에 있을 수 있으므로 직접 연결
    public Slider boostSlider;                  // UI 슬라이더는 비워두면 자동으로 생성됩니다.

    [Header("이동 속도")]
    public float moveSpeedMultiplier = 1.0f;
    
    [Header("점프 설정")]
    public float normalJumpHeight = 1.0f;
    public float maxJumpBonus = 1.5f;
    public float gravity = -9.81f;

    [Header("부스트 설정")]
    public float maxBoostGauge = 100f;
    public float boostFillRate = 10f;
    public float boostDrainRate = 25f;
    public float penaltyDrainRate = 5f;
    public float boostSpeedMultiplier = 2.0f;
    [Range(-10, 60)]
    public float optimalPitchMin = 10f;
    [Range(-10, 60)]
    public float optimalPitchMax = 30f;

    private CharacterController controller;
    private Animator animator;

    private Vector3 playerVelocity;
    private bool isGrounded;
    private float currentBoostGauge;

    void Start()
    {
        // 게임이 시작될 때 컴포넌트를 자동으로 찾아옵니다.
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
        currentBoostGauge = 0f;

        // ✨ [핵심 수정] boostSlider가 인스펙터에서 연결되지 않았다면, 직접 생성합니다.
        if (boostSlider == null)
        {
            Debug.Log("Boost Slider가 연결되지 않아 자동으로 생성합니다.");
            SetupBoostUI();
        }
    }

    void Update()
    {
        isGrounded = controller.isGrounded;
        if (isGrounded && playerVelocity.y < 0)
        {
            playerVelocity.y = -2f;
        }

        if (kneeController == null) return;
        
        float kneePitch = kneeController.CurrentPitch;
        bool isBoosting = Input.GetKey(KeyCode.LeftShift) && currentBoostGauge > 0;

        if (!isBoosting)
        {
            if (kneePitch >= optimalPitchMin && kneePitch <= optimalPitchMax)
            {
                currentBoostGauge += boostFillRate * Time.deltaTime;
            }
            else
            {
                currentBoostGauge -= penaltyDrainRate * Time.deltaTime;
            }
        }
        
        float currentSpeedMultiplier = moveSpeedMultiplier;
        if (isBoosting)
        {
            currentBoostGauge -= boostDrainRate * Time.deltaTime;
            currentSpeedMultiplier *= boostSpeedMultiplier;
        }
        currentBoostGauge = Mathf.Clamp(currentBoostGauge, 0, maxBoostGauge);
        
        float animSpeed = animator.GetFloat("Speed");
        Vector3 move = transform.forward * animSpeed * currentSpeedMultiplier * Time.deltaTime;
        controller.Move(move);

        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            float bendFactor = Mathf.InverseLerp(kneeController.maxPitch, kneeController.minPitch, kneePitch);
            float jumpBonus = bendFactor * maxJumpBonus;
            float finalJumpHeight = normalJumpHeight + jumpBonus;
            playerVelocity.y = Mathf.Sqrt(finalJumpHeight * -2f * gravity);
        }

        playerVelocity.y += gravity * Time.deltaTime;
        controller.Move(playerVelocity * Time.deltaTime);

        if (boostSlider != null)
        {
            boostSlider.value = currentBoostGauge / maxBoostGauge;
        }
    }

    // ✨ [추가] UI를 자동으로 생성하고 설정하는 전체 함수
    void SetupBoostUI()
    {
        CreateCanvasAndEventSystem();

        Canvas canvas = FindObjectOfType<Canvas>();

        GameObject sliderObj = new GameObject("BoostGaugeUI");
        sliderObj.transform.SetParent(canvas.transform, false);

        RectTransform sliderRect = sliderObj.AddComponent<RectTransform>();
        sliderRect.anchorMin = new Vector2(0, 0);
        sliderRect.anchorMax = new Vector2(0, 0);
        sliderRect.pivot = new Vector2(0, 0);
        sliderRect.anchoredPosition = new Vector2(30, 30);
        sliderRect.sizeDelta = new Vector2(250, 20);

        GameObject backgroundObj = new GameObject("Background");
        backgroundObj.transform.SetParent(sliderObj.transform, false);
        Image backgroundImage = backgroundObj.AddComponent<Image>();
        backgroundImage.color = new Color(0.1f, 0.1f, 0.1f, 0.6f);
        RectTransform backgroundRect = backgroundObj.GetComponent<RectTransform>();
        backgroundRect.anchorMin = Vector2.zero;
        backgroundRect.anchorMax = Vector2.one;
        backgroundRect.sizeDelta = Vector2.zero;

        GameObject fillAreaObj = new GameObject("Fill Area");
        fillAreaObj.transform.SetParent(sliderObj.transform, false);
        RectTransform fillAreaRect = fillAreaObj.AddComponent<RectTransform>();
        fillAreaRect.anchorMin = Vector2.zero;
        fillAreaRect.anchorMax = Vector2.one;
        fillAreaRect.sizeDelta = Vector2.zero;

        GameObject fillObj = new GameObject("Fill");
        fillObj.transform.SetParent(fillAreaObj.transform, false);
        Image fillImage = fillObj.AddComponent<Image>();
        fillImage.color = new Color(0.2f, 0.7f, 1f, 1f);

        Slider newSlider = sliderObj.AddComponent<Slider>();
        newSlider.fillRect = fillObj.GetComponent<RectTransform>();
        newSlider.direction = Slider.Direction.LeftToRight;
        newSlider.minValue = 0;
        newSlider.maxValue = 1;
        newSlider.value = 0;
        newSlider.handleRect = null;

        this.boostSlider = newSlider;
    }

    void CreateCanvasAndEventSystem()
    {
        if (FindObjectOfType<EventSystem>() == null)
        {
            GameObject eventSystemObj = new GameObject("EventSystem");
            eventSystemObj.AddComponent<EventSystem>();
            eventSystemObj.AddComponent<StandaloneInputModule>();
        }
        if (FindObjectOfType<Canvas>() == null)
        {
            GameObject canvasObj = new GameObject("Canvas");
            Canvas canvas = canvasObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasObj.AddComponent<CanvasScaler>();
            canvasObj.AddComponent<GraphicRaycaster>();
        }
    }
}
