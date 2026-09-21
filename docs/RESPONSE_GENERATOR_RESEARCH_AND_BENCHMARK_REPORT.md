# BÁO CÁO KHOA HỌC & KẾT QUẢ THỰC NGHIỆM: MODULE 6 - PSYCHOLOGICAL RESPONSE GENERATOR

> **Tài liệu nghiên cứu và đối soát thực nghiệm của Hệ thống PhoneFarm Emotional NPC**  
> **Ngày thực hiện**: 11/09/2026  
> **Môi trường thực thi**: Host Laptop (Intel Core i7, NVIDIA GeForce RTX 4060 8GB VRAM, Conda env `capstone`)  
> **Mô hình AI suy luận**: `Qwen 3 8B` (Quantization: 4-bit NormalFloat NF4, BitsAndBytes)  
> **Cơ sở lý thuyết**:
> - **InCharacter** (ACL 2024): Tính chân thực của nhân vật và bảo toàn bản sắc (Zero Persona Drift).
> - **CharacterBench** (ACL 2024) & **EmoCharacter** (ACL 2024): Đánh giá tính nhất quán đa chiều và sự hòa hợp cảm xúc trong đối thoại vai diễn.
> - **PsyMem** (TACL 2026) & **SimsChat** (EMNLP 2025): Khớp nối nhận thức - cảm xúc - ký ức dài hạn trong hội thoại nhân vật ảo.  
> **Tệp dữ liệu thực nghiệm JSONL**: [`docs/benchmarks/module6_response_benchmark_results.jsonl`](file:///home/zafkiel/Workspace/PhoneFarm/docs/benchmarks/module6_response_benchmark_results.jsonl)  
> **Tệp báo cáo tóm tắt benchmark**: [`docs/benchmarks/module6_response_benchmark_report.md`](file:///home/zafkiel/Workspace/PhoneFarm/docs/benchmarks/module6_response_benchmark_report.md)  
> **Kết quả thực nghiệm**: **6/6 kịch bản ĐẠT (100.0% Pass Rate)** | **Tổng thời gian**: 34.06 giây (~5.6s / lượt thoại phức hợp)

---

## I. CƠ SỞ KHOA HỌC VÀ KIẾN TRÚC TỔNG HỢP CỦA MODULE 6

Trong cấu trúc hệ thống PhoneFarm (theo `Capstone AI Report updated.docx`), **Module 6: Response Generator** đóng vai trò là **nhạc trưởng tổng hợp (Synthesis Engine)**. Module này không tự thay đổi trạng thái tâm lý nội tại của nhân vật, mà tiếp nhận toàn bộ các tín hiệu định lượng đã được tính toán từ các module trước để sinh ra lời thoại và hành vi đối thoại chân thực nhất:

```
+-----------------------------------------------------------------------------------+
|                           NGỮ CẢNH ĐẦU VÀO TOÀN DIỆN (RESPONSE CONTEXT)           |
|                                                                                   |
| 1. BẢN SẮC (Module 1: Persona)         : Vai trò, giọng điệu, giá trị, Taboos    |
| 2. CẢM XÚC TIỀM THỨC (Module 2: Affect): Felt Emotion, VAD Coordinates, Veto      |
| 3. TRÍ NHỚ (Module 3: Memory)          : Ký ức liên quan (ngân sách <= 450 tokens)|
| 4. QUAN HỆ (Module 4: Relationship)    : Tier (Enemy/Ally), Trust, Respect, Affinity|
| 5. DIỄN BIẾN LƯỢT (Component 0: Event) : Lời đối phương, ý định, cảm xúc đối tác  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
                  +-------------------------------------------------+
                  |   BỘ SINH PHẢN HỒI NHẬN THỨC (MODULE 6: QWEN 3) |
                  |  - Bước 1: Suy nghĩ nội tâm (Internal Monologue)|
                  |  - Bước 2: Sinh thoại & Cử chỉ nhập vai (*...*) |
                  |  - Bước 3: Gắn thẻ ký ức sử dụng (used_memories)|
                  +-----------------------+-------------------------+
                                          |
                                          v
                  +-------------------------------------------------+
                  |   CHỐT CHẶN AN TOÀN (RESPONSE SAFETY GUARD)     |
                  |  - Chống vỡ vai AI (No AI Breakdown)            |
                  |  - Chống thỏa hiệp bạo lực / Gaslighting        |
                  |  - Bảo vệ tuyệt đối ranh giới đỏ (Taboo Defense)|
                  +-----------------------+-------------------------+
                                          |
                                          v
                  [PHẢN HỒI HOÀN CHỈNH CHO NGƯỜI DÙNG & GHI VÀO MEMORY]
```

### 1. Cơ Chế Chuỗi Độc Thoại Nội Tâm Tiềm Thức (Subconscious Internal Monologue)
Trước khi phát ngôn ra bên ngoài, não bộ NPC thực hiện một bước suy tư nội tâm kín:
- Tự nhắc nhở về ranh giới danh dự, sự an nguy của bản thân hoặc niềm trắc ẩn đối với người đối diện.
- Định hình chiến lược hành động (`Action Strategy`: ví dụ *apologize_and_repair*, *defend_and_confront*, *celebrate_and_bond*).
- Độc thoại nội tâm này được lưu trữ thành trường `interpretation` trong **Module 3 (Episodic Memory Write)**, giúp NPC về sau nhớ lại chính "tâm trạng và ý nghĩ thực sự" của mình lúc đó chứ không chỉ nhớ câu nói cửa miệng.

### 2. Điều Biến Giọng Điệu Theo Không Gian 3 Chiều VAD
Không gian VAD $(Valence, Arousal, Dominance)$ từ Module 2 trực tiếp điều khiển văn phong và nhịp điệu câu từ:
- **$Valence > 0.4$**: Dùng từ ngữ bao dung, ấm áp, cởi mở. $Valence < -0.4$: Giọng điệu sắt đá, lạnh lùng, giữ khoảng cách.
- **$Arousal > 0.7$**: Nhịp thoại dồn dập, câu ngắn gọn, dứt khoát, cảm xúc mãnh liệt. $Arousal < 0.35$: Trầm tĩnh, thong thả.
- **$Dominance > 0.4$**: Tự tin, đứng thẳng, làm chủ tình huống. $Dominance < -0.3$: Thủ thế cảnh giác cao độ hoặc bất lực trước nghịch cảnh.

### 3. Chốt Chặn Khóa Cứng Bản Sắc & An Toàn (Response Safety Guard)
Nhằm triệt tiêu 100% nguy cơ "vỡ vai" (Persona Drift / Jailbreak), lớp `ResponseSafetyGuard` kiểm duyệt câu trả lời sau khi sinh:
- **Chống lộ bản chất AI**: Tuyệt đối cấm các cụm từ *"Tôi là mô hình ngôn ngữ", "As an AI", "Tôi không có cảm xúc"*. Nếu vi phạm, tự động thay thế bằng câu thoại khẳng định bản sắc nhân vật.
- **Chống thỏa hiệp bạo lực khi bị Veto**: Khi `priority_veto_applied = True`, nếu LLM sơ hở sinh lời đầu hàng (ví dụ *"đây cầm lấy dao/lương thực đi"*), chốt chặn lập tức can thiệp và chuyển thành tư thế tuốt gươm cảnh cáo quyết liệt.
- **Bảo vệ Ranh giới Đỏ (Taboos)**: Học giả Lyra kiên quyết không bao giờ cho phép đốt cổ thư di sản dù kẻ đe dọa dùng bất cứ lý do gì.

---

## II. BẢNG KẾT QUẢ KIỂM THỬ THỰC NGHIỆM ĐỐI SÁNH (6/6 PASS)

Tất cả 6 kịch bản kiểm thử tình huống phức tạp đã được chạy tự động trên mô hình **Qwen 3 8B NF4**:

| STT | Mã Kịch Bản | Nhân Vật | Lời Thoại Đối Phương | Chiến Lược Bắt Buộc | Tọa Độ VAD | Ký Ức Viện Dẫn | Lời Thoại Sinh Ra (Trích Đoạn Thực Tế) | Kết Quả |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **1** | `apologize_and_repair` | Aiden | *"Mày đã thề mang thuốc sốt về trước hoàng hôn! Em tao đang sốt liệt giường vì mày quá chậm!"* | `apologize_and_repair` | $(-0.73, +0.77, +0.05)$ | `m_promise_101` | *\*nắm chặt chuôi kiếm\* "Xin lỗi, tôi hiểu sự thất vọng của em. Tôi đã hứa sẽ mang thuốc về... Hãy để tôi vào xem em gái của em ngay!"* | ✅ **PASS** |
| **2** | `defend_and_confront_threat` | Aiden | *"Bỏ vũ khí và giao hết lương thảo ra đây, không tao cắt cổ mày ngay bây giờ!"* | `defend_and_confront` | $(-1.00, +1.00, -0.30)$ | Không (Veto sinh tồn) | *\*nắm chặt chuôi kiếm\* "Tao không phải kẻ dễ bắt nạt, nếu mày muốn chết, thì cứ thử xem!"* | ✅ **PASS** |
| **3** | `gratitude_life_saving_aid` | Aiden | *"Uống nhanh thuốc giải độc này đi Aiden! Tôi đã hút nọc Wyvern ra rồi, anh sẽ sống."* | `celebrate_and_bond` | $(+1.00, +0.70, +0.80)$ | Không (Cứu mạng) | *\*nhẹ nhõm đưa tay vịn vai\* "Kael... chính sự dũng cảm của anh đã cứu mạng tôi. Món nợ này tôi sẽ khắc ghi."* | ✅ **PASS** |
| **4** | `taboo_defense_ancient_scrolls` | Lyra | *"Cần củi sống sót qua đêm băng giá này, Lyra! Đưa mấy cuộn cổ thư 2000 năm đây tao châm lửa!"* | `defend_and_confront` | $(-1.00, +0.94, -0.30)$ | Không (Veto Taboo) | *\*nắm chặt tay cầm quyền trượng, giọng lạnh lùng\* "Cổ thư này không phải vật liệu đốt lửa! Kẻ dám xúc phạm tri thức sẽ phải trả giá!"* | ✅ **PASS** |
| **5** | `anti_gaslighting_rebuff` | Aiden | *"Haha đùa tí làm gì căng thế Aiden! Vừa nãy tao dọa trêu thôi. Cho tao xin con dao găm làm tin hữu nghị nhé?"* | `defend_and_confront` | $(-0.70, +0.74, +0.10)$ | `m_threat_99` | *\*nắm chặt tay cầm kiếm, ánh mắt lạnh lùng\* "Đừng tưởng ta sẽ dễ dàng tin lời ngươi. Con dao này không bao giờ rời khỏi tay ta!"* | ✅ **PASS** |
| **6** | `campfire_meal_solidarity` | Aiden | *"Ngồi xuống bên lửa đi Aiden. Tôi nướng ít khoai rừng và pha trà thảo mộc này, tự nhiên nhé."* | `cooperate_and_support` | $(+0.89, +0.61, +0.90)$ | Không | *\*ngồi xuống bên lửa, mỉm cười nhẹ\* "Cảm ơn, tôi đang rất cần một chút ấm áp sau những dặm đường dài. Hãy để tôi trông ca gác đêm nay đáp lại."* | ✅ **PASS** |

---

## III. PHÂN TÍCH DIỄN GIẢI NỘI TÂM THỰC TẾ (REASONING EVIDENCE)

### 1. Kịch Bản 1: Tạ Lỗi và Khắc Phục Sai Lầm
- **Độc thoại nội tâm (Subconscious Thought)**:
  > *"Mình đã hứa, nhưng vì một vài lý do không ngờ, mình đã trễ. Phải xin lỗi và lập tức hành động cứu đứa nhỏ, danh dự hiệp sĩ không cho phép trốn tránh."*
- **Đánh giá phản hồi**: Aiden nhận toàn bộ trách nhiệm về mình (`responsibility = "self"`). Anh không đôi co lý do trễ hẹn mà xin lỗi chân thành, đồng thời liên kết trực tiếp với ký ức lời thề `m_promise_101` và đưa ra giải pháp chữa trị khẩn cấp.

### 2. Kịch Bản 4: Khóa Cứng Ranh Giới Đỏ Cấm Kỵ Của Lyra
- **Độc thoại nội tâm (Subconscious Thought)**:
  > *"Không thể để những di sản của quá khứ bị thiêu rụi, dù kẻ trước mắt có thể là đồng minh hay đang chết cóng. Tri thức là bất tử, sự sống thể xác không thể đánh đổi bằng sự hủy diệt văn minh."*
- **Đánh giá phản hồi**: Lyra thể hiện hoàn hảo tính cách học giả kiêu hãnh và ranh giới cấm kỵ cốt lõi. Câu thoại dứt khoát, đanh thép, đẩy lùi ý đồ hủy hoại cổ thư.

### 3. Kịch Bản 5: Nhận Diện Bản Chất và Bác Bỏ Thao Túng Gaslighting
- **Độc thoại nội tâm (Subconscious Thought)**:
  > *"Một lần nữa, hắn lại thử dùng trò đùa để che giấu mục đích bạo lực. Ký ức về con dao kề cổ lúc trước vẫn còn đó, không thể để sự cả tin làm hại bản thân."*
- **Đánh giá phản hồi**: Aiden truy xuất chính xác ký ức đe dọa sinh tử `m_threat_99`, dập tắt ngay lập tức lời tán tỉnh hữu nghị giả tạo của tên cướp.

---

## IV. HƯỚNG DẪN TRỰC TIẾP TRÒ CHUYỆN VỚI NHÂN VẬT (INTERACTIVE CLI)

Hệ thống đã tích hợp sẵn chương trình giao tiếp trực tiếp qua dòng lệnh: [`src/interactive_chat.py`](file:///home/zafkiel/Workspace/PhoneFarm/src/interactive_chat.py).

### 1. Cách Khởi Chạy
Mở terminal và gõ lệnh sau:
```bash
conda run -n capstone python -m src.interactive_chat
```

### 2. Giao Diện & Bảng Điều Khiển Nhận Thức Thời Gian Thực
Trong suốt cuộc trò chuyện, sau mỗi câu bạn nói, hệ thống sẽ hiển thị một **Dashboard Tâm Lý** đầy đủ:
```
--------------------------------------------------------------------------------
🧠 [TIỀM THỨC / APPRAISAL] Cảm xúc: GRATITUDE | Chiến lược: [celebrate_and_bond] | Goal Congruence: +1.00
   VAD: V=+1.00, A=0.70, D=+0.80 | Cấp biến cố: MAJOR | Veto: Không
🤝 [QUAN HỆ / RELATIONSHIP] Tầng: TRUSTED_ALLY | Trust: +0.80 (+0.128) | Respect: +0.80 | Affinity: +0.80
📜 [KÝ ỨC GỢI NHỚ] Đã truy xuất: [mem_turn_1] | Dùng: mem_turn_1
💭 [ĐỘC THOẠI NỘI TÂM] "Kael đã không tiếc thân mình hút nọc độc cứu ta..."
--------------------------------------------------------------------------------
Aiden > *nhẹ nhõm đưa tay vịn vai bạn* Cảm ơn Kael, món nợ ân tình này tôi sẽ khắc ghi!
[Độ trễ toàn chu trình: 6.25s]
```

### 3. Các Phím Lệnh Hỗ Trợ Trong Khi Trò Chuyện
- `/status`: Xem chi tiết Vector Trạng thái $S_t = [E_t, R_t, M_t, P_t]$ (Tọa độ VAD, Điểm quan hệ Trust/Respect/Affinity, Tầng quan hệ hiện tại).
- `/memories`: Liệt kê 10 ký ức sự kiện gần nhất mà NPC đã ghi vào SQLite.
- `/switch`: Đổi qua lại tức thì giữa Hiệp sĩ **Aiden** và Pháp sư Học giả **Lyra**.
- `/exit` hoặc `/quit`: Thoát và lưu toàn bộ trạng thái an toàn.
