# PHONEFARM COGNITIVE PIPELINE: HỢP ĐỒNG KIẾN TRÚC & CÁC QUY TẮC BẤT BIẾN (INVARIANTS)
**Dự án**: Capstone PhoneFarm Distributed Emotional AI NPC  
**Trọng tâm cốt lõi**: Song trụ Nhận thức **PERSONA (Bản ngã)** & **MEMORY (Ký ức)**  
**Trạng thái**: Chuẩn mực ràng buộc toàn hệ thống (System-wide Architecture Contract)

---

## 1. NGUYÊN LÝ SONG TRỤ NHẬN THỨC (THE DUAL PILLARS OF COGNITION)

Trong tâm lý học nhận thức và kiến trúc tác nhân PhoneFarm, trạng thái tâm lý toàn vẹn được hình thức hóa thành bộ tứ:
$$S_t^{(i, j)} = \left[ E_t^{(i)}, R_t^{(i, j)}, M_t^{(i)}, P_t^{(i)} \right]$$

Hệ thống đứng vững trên **HAI TRỤ CỘT NỀN MÓNG CÓ TẦM QUAN TRỌNG NGANG NHAU**:
1. **MEMORY ($M_t$ - Trải nghiệm sống)**: *"Những gì đã xảy ra với tôi và tôi đã cảm nhận nó thế nào trong quá khứ?"* (Autobiographical History).
2. **PERSONA ($P_t$ - Bản ngã & Lăng kính nhận thức)**: *"Tôi là ai, tôi coi trọng điều gì, và tôi nhìn nhận thế giới qua lăng kính nào?"* (Soul & Moral Framework).

> ⚠️ **Quy luật tương hỗ**: 
> - Ký ức ($M_t$) nếu không có Bản sắc ($P_t$) soi chiếu sẽ chỉ là một đống văn bản vô hồn; cùng một câu mắng, kẻ hèn nhát thì sợ hãi, còn dũng sĩ thì phẫn nộ.
> - Bản sắc ($P_t$) nếu không có Ký ức ($M_t$) tích lũy sẽ mãi mãi dậm chân tại chỗ, trở thành một chatbot vô cảm không có khả năng phát triển cung nhân vật (Flat Tone & Context Amnesia).

---

## 2. BỐN TẦNG CẤU TRÚC CỦA PERSONA ($P_t$)

Theo thiết kế trong [`src/module1_persona/schema.py`](file:///home/zafkiel/Workspace/PhoneFarm/src/module1_persona/schema.py) và [`Capstone AI Report updated.docx`](file:///home/zafkiel/Workspace/PhoneFarm/Capstone%20AI%20Report%20updated.docx):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│   TẦNG 1: IDENTITY & TABOOS (Khóa cứng 100% - Bất biến tuyệt đối)                      │
│   • Tên, tuổi, giới tính, hình mẫu (Archetype), xuất thân lịch sử.                     │
│   • Ranh giới đỏ sinh tử (Taboos): Tuyệt đối không phục tùng áp bức, không bán rẻ bạn. │
├────────────────────────────────────────────────────────────────────────────────────────┤
│   TẦNG 2: BIG FIVE TRAITS (OCEAN ∈ [0.0, 1.0]^5 - Trạng thái chậm)                     │
│   • Openness (Mở rộng), Conscientiousness (Tận tâm), Extraversion (Hướng ngoại),       │
│     Agreeableness (Dễ chịu / Tin người), Neuroticism (Bất ổn cảm xúc).                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│   TẦNG 3: VALUES & WORLDVIEW (Giá trị sống & Thế giới quan - Trạng thái tiến hóa)      │
│   • Thang đo giá trị Schwartz: Tự do, Tri thức, Trung thành, Công lý, Sinh tồn.       │
│   • Niềm tin xã hội: Lòng tin con người (Trust Worldview), Thái độ với quyền lực.      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│   TẦNG 4: DYNAMIC STATUS (Thể chất & Cảm xúc tức thời - Trạng thái nhanh)              │
│   • Máu (Health), Thể lực (Energy), Mức căng thẳng (Stress Level).                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MA TRẬN RÀNG BUỘC TOÀN DIỆN: PERSONA & MEMORY LÊN TOÀN BỘ CÁC MODULE

Bảng dưới đây quy định bắt buộc **tác động 2 chiều** của cả **Persona** và **Memory** lên từng chặng trong chu trình xử lý:

| Chặng / Module | Ràng buộc từ PERSONA ($P_t$) | Ràng buộc từ MEMORY ($M_t$) | Tác động ngược lại |
| :--- | :--- | :--- | :--- |
| **Module 0: Event Interpreter** | Xác định ngôi xưng, quan hệ actor-target (Aiden hay Lyra là mục tiêu). | Cung cấp ngữ cảnh 2 lượt trước để giải quyết đại từ thay thế (*"nó", "chuyện đó"*). | Tạo ra `EventContext` sạch (sự kiện khách quan + nguyên nhân). |
| **Module 2: Cognitive Appraisal** | **LĂNG KÍNH THẨM ĐỊNH**: <br>• Giá trị sống & Mục tiêu quyết định `goal_congruence`.<br>• Ranh giới đỏ (Taboos) quyết định `norm_compatibility`.<br>• Neuroticism & Agreeableness quyết định cường độ giận dữ hay tha thứ. | **CƠ SỞ THAM CHIẾU**: <br>• Đọc Top-k ký ức liên quan để biết đối phương từng là ân nhân hay kẻ thù nhằm xác định đúng `responsibility` và mức độ nghiêm trọng. | **Sinh ra 4 trường sinh tử cho Memory**:<br>`felt_emotion`, `interpretation`, `salience`, `severity` (*minor, major, trauma*). |
| **Module 3: Episodic Memory (Pha Đọc & Ghi)** | Dùng Values & Goals để tính độ liên quan cá nhân $\text{Relevance}(e, P_t) \implies \text{Salience} = \|V\| \cdot A \cdot Rel$. | **Tự vận hành ACT-R Decay**: <br>• Ký ức chấn thương vi phạm Taboos có $d_{\text{eff}} = 0.15$ (Kháng suy giảm).<br>• Ràng buộc Context Budget $\le 450$ tokens. | Lưu trữ bản ghi SQLite đầy đủ nhãn tâm lý. |
| **Module 4: Dynamic Relationship** | Persona quyết định **Lòng tin cơ sở (Baseline Trust)**: <br>• Aiden thận trọng: $0.50$<br>• Lyra cởi mở: $0.65$<br>• Kẻ hoài nghi: $0.20$. | Cung cấp lịch sử số lần thất hứa hoặc giúp đỡ trong quá khứ để tính hệ số leo thang (`escalation_factor`). | Cập nhật $\Delta Trust, \Delta Respect, \Delta Affection$ và đính kèm vào bản ghi Memory. |
| **Module 6: Response Generator** | **ĐỊNH HÌNH GIỌNG ĐIỆU & CÂU CHỮ**: <br>• Giữ đúng Speaking Style, đại từ nhân xưng.<br>• Tuân thủ nghiêm ngặt behavioral constraints (Aiden không van xin, Lyra dùng thuật ngữ học giả). | **CĂN CỨ SỰ KIỆN NÓI NĂNG**: <br>• Nhắc lại đúng chi tiết quá khứ (ví dụ: vết thương vai, thanh kiếm thiên thạch), không bịa đặt ký ức giả. | Sinh ra `agent_response` thực tế để lưu vào Memory. |
| **Module 5: Character Evolution** | **ĐỐI TƯỢNG ĐƯỢC TIẾN HÓA**: <br>• Chỉ Worldview Beliefs và Big Five được thay đổi.<br>• `Identity` bị khóa cứng vĩnh viễn. | **NGUỒN NĂNG LƯỢNG BẰNG CHỨNG**: <br>• Quét toàn bộ ký ức chấn thương để tính Cổng bão hòa $\tanh(\text{Raw}/\beta) \ge \theta_P$. | Ghi mẩu ký ức `[CHIÊM NGHIỆM SÂU SẮC]` vào Memory và giải phóng Cổng bằng chứng về 0. |

---

## 4. BẢY QUY TẮC BẤT BIẾN TOÀN HỆ THỐNG (SYSTEM INVARIANTS)

1. **Bất biến 1 (Khóa cứng Bản sắc - Immutable Identity & Zero Persona Drift)**:
   Dù trải qua hàng nghìn biến cố hay bị người dùng cố tình "dụ dỗ/jailbreak", nhân vật **tuyệt đối không bao giờ được phép từ bỏ ranh giới đỏ (Taboos), xuất thân, và quy tắc đạo đức bất biến** đã khai báo trong `Identity`.
2. **Bất biến 2 (Thẩm định Nhận thức phụ thuộc Bản sắc - Persona-grounded Appraisal)**:
   Module 2 không được phép gán nhãn cảm xúc theo cảm tính chung chung. Mọi chỉ số `goal_congruence` và `norm_compatibility` bắt buộc phải được giải thích dựa trên chính `Values` và `Taboos` của nhân vật đang kích hoạt.
3. **Bất biến 3 (Giới hạn Ngân sách Trí nhớ Làm việc - Working Memory Ceiling)**:
   Dù kho SQLite có 10.000 ký ức, số lượng ký ức được đưa vào Context Prompt của LLM tại mỗi lượt **tuyệt đối không vượt quá 8 mục (ngân sách $\le 450$ tokens)** để bảo vệ mô hình Qwen 3 8B khỏi hiện tượng phân tán chú ý.
4. **Bất biến 4 (Ghi Ký ức Nguyên tử - Atomic Memory Ingestion)**:
   Sau mỗi câu thoại của người dùng mà NPC có phản hồi, kho ký ức **bắt buộc phải tăng chính xác thêm 1 bản ghi** kèm đầy đủ vector nhúng 384 chiều, cảm xúc và diễn giải tâm lý. Không lượt tương tác nào được phép kết thúc nếu chưa có `committed_memory_id`.
5. **Bất biến 5 (Suy giảm Vi sai Đèn Flash - Flashbulb Differential Decay)**:
   Các ký ức mang cấp độ `[TRAUMA]` hoặc có $\text{Salience} \ge 0.70$ (chạm vào giá trị cốt lõi của Persona) bắt buộc phải sử dụng hệ số suy giảm thời gian chậm $d_{\text{eff}} = 0.15$ thay vì $0.50$.
6. **Bất biến 6 (Tiến hóa Tính cách có Kiểm soát Ngưỡng - Saturated Gate Evolution)**:
   Tính cách và Thế giới quan của Persona **chỉ được phép thay đổi khi và chỉ khi** tổng điểm bằng chứng bão hòa $\tanh(\text{Evidence\_Raw} / \beta)$ vượt ngưỡng thích ứng $\theta_P$. Không bao giờ được phép đổi tính sau 1 vài câu thoại rời rạc.
7. **Bất biến 7 (Cô lập Tuyệt đối Đa Nhân vật - Multi-Agent Partitioning)**:
   Mọi thao tác đọc/ghi vào Memory hoặc cập nhật Persona bắt buộc phải có khóa `agent_id`, bảo đảm không bao giờ có hiện tượng rò rỉ ký ức hay lai tạp tính cách giữa Aiden và Lyra.

---

## 5. CẤU TRÚC ĐIỀU PHỐI MÃ NGUỒN (`TurnContext` State Machine)

Khi hiện thực hóa `src/pipeline/turn_context.py`, luồng dữ liệu sẽ ép buộc kiểm tra sự hiện diện của cả Persona và Memory:

```python
@dataclass
class TurnContext:
    # Đầu vào
    raw_message: str
    actor_id: str
    character_id: str                   # "aiden" hoặc "lyra"
    timestamp: float
    
    # Hai trụ cột nền tảng
    active_persona: CharacterProfile    # Module 1 (Bản ngã & Taboos)
    retrieved_memories: list[Memory]    # Module 3 Phase Đọc (Quá khứ liên quan)
    
    # Các chặng suy luận
    event_context: EventContext         # Component 0
    appraisal_result: AppraisalResult   # Module 2 (Thẩm định kết hợp Persona + Memory)
    relationship_state: RelState        # Module 4
    response_text: str                  # Module 6
    
    # Chốt chặn cuối lượt
    committed_memory_id: str            # Module 3 Phase Ghi (Bắt buộc != None)
    evolution_patch: dict | None = None # Module 5 (Nếu vượt ngưỡng tanh)
```
