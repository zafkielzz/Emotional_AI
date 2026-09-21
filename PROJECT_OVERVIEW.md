# PROJECT OVERVIEW: TRACEABLE EMOTIONAL AI & DYNAMIC CHARACTER ARCS IN A SANDBOX ON ASYMMETRIC EDGE-HOST ARCHITECTURE (V2)

---

## 1. TỔNG QUAN & TẦM NHÌN DỰ ÁN (PROJECT VISION)

Dự án phát triển một khung kiến trúc tác nhân nhận thức (**Cognitive-Affective Agent Architecture**) cho các nhân vật không thể điều khiển (NPC) trong môi trường mô phỏng thế giới ảo (Sandbox) với khả năng tương tác ngôn ngữ tự do (Open-ended Dialogue).

Hệ thống giải quyết bài toán: **Làm thế nào để NPC phát triển trạng thái cảm xúc, mối quan hệ cặp đôi và tiến hóa tính cách dài hạn một cách nhất quán, có thể truy vết (traceable) từ chuỗi trải nghiệm sống mà không bị trôi dạt nhân vật (zero character drift)?**

Hệ thống vận hành trên kiến trúc tính toán bất đối xứng (**Asymmetric Edge-Host Architecture**):
- 📱 **Điện thoại di động (Phone Edge Node)**: Chạy mô hình ngôn ngữ nhỏ (SLM 0.5B – 1.5B) phục vụ **sinh câu thoại nhập vai (Dialogue Generation)** theo vector trạng thái đã được định hình và phân loại cảm xúc bề mặt (Surface Emotion Detection).
- 💻 **Máy tính trung tâm (Host Laptop GPU RTX 4060)**: Chạy mô hình lớn hơn (LLM 8B) đảm nhiệm **Thẩm định nhận thức sâu (Deep Cognitive Appraisal)**, tính toán **Cổng bằng chứng bão hòa (Saturated Evidence Gate)**, điều phối **Phân xử xung đột (Conflict Resolution)**, thực hiện **Tự phản tư sâu (Deep Reflection)** và quản lý thế giới ảo (World Controller).

> ⚠️ **Tuyên bố Giới hạn Khoa học (Scientific Disclaimer)**:
> Dự án sử dụng thuật ngữ **"Traceable"** (Có thể truy vết) thay vì **"Causal"** (Nhân quả) theo nghĩa thống kê can thiệp hay phản thực (Pearl, 2009). Mọi sự chuyển dịch trạng thái của nhân vật đều có thể truy nguyên giải thích (Explainable & Auditable) về các biến cố cụ thể trong quá khứ, nhưng không giả định một mô hình nhân quả cấu trúc (SCM) hoàn chỉnh. Đồng thời, thuật ngữ **"Dynamic Character Arc" / "Self-organized Trajectory"** được sử dụng thay cho **"Emergent"** để phản ánh chính xác cơ chế tự tổ chức trạng thái có kiểm soát bằng ngưỡng toán học.

---

## 2. FORMAL STATE TRANSITION MODEL (HỆ TOÁN HỌC CHUYỂN TRẠNG THÁI)

### 2.1. Không gian Trạng thái (State Representation)
Tại mỗi bước thời gian $t$, toàn bộ trạng thái tâm lý nội tại của NPC $i$ đối với thực thể $j$ được hình thức hóa thành bộ tứ trạng thái:
$$S_t^{(i, j)} = \left[ E_t^{(i)}, R_t^{(i, j)}, M_t^{(i)}, P_t^{(i)} \right]$$

Trong đó:
1. **$E_t^{(i)} \in [-1.0, 1.0]^k$**: Vector cảm xúc vi mô (Fast State), gồm các chiều liên tục Valence, Arousal, Dominance (VAD) và các cảm xúc rời rạc (Anger, Fear, Sadness, Joy).
2. **$R_t^{(i, j)} \in [-1.0, 1.0]^3$**: Vector quan hệ hai chiều (Dyadic Relationship - Medium State) giữa $i$ và $j$, gồm $[\text{Trust}, \text{Affinity}, \text{Respect}]$.
3. **$M_t^{(i)}$**: Tập hợp các mẩu ký ức sự kiện chủ quan (Episodic Memories) kèm nhãn cảm xúc và dấu thời gian.
4. **$P_t^{(i)}$**: Bộ tham số nhân cách cốt lõi (Slow State), gồm 5 nét tính cách Big Five $\in [0.0, 1.0]^5$, Hệ giá trị Schwartz $\in [0.0, 1.0]^m$ và Thế giới quan (Worldview Beliefs).

---

### 2.2. Hình thức hóa Cổng Bằng chứng Chuẩn hóa & Bão hòa (Saturated Evidence Gate)

Để ngăn chặn hiện tượng **tràn số (Unbounded Growth)** khi NPC tích lũy hàng nghìn sự kiện nhỏ trong suốt vòng đời mô phỏng, tổng điểm bằng chứng thô (Raw Evidence) được chuẩn hóa về khoảng $[0.0, 1.0)$ bằng hàm tiếp tuyến hyperbolic ($\tanh$):

$$\text{Evidence\_Raw}_t(i) = \sum_{j=1}^{|W|} \left[ w_j \cdot \text{Salience}(e_j) \cdot \text{Recency}(e_j) \cdot \text{Repetition}(e_j) \right]$$

$$\text{Evidence\_Normalized}_t(i) = \tanh\left( \frac{\text{Evidence\_Raw}_t(i)}{\beta} \right)$$

Trong đó:
- **$\beta$**: Hệ số tỷ lệ (scaling factor, mặc định $\beta = 2.0$) điều chỉnh độ dốc bão hòa.
- **$w_j$**: Trọng số cấp độ biến cố ($w_{\text{Minor}} = 0.3, w_{\text{Major}} = 1.0, w_{\text{Trauma}} = 2.5$).
- **$\text{Salience}(e_j) = |\text{Valence}(e_j)| \cdot \text{Arousal}(e_j) \cdot \text{Relevance}(e_j, P_t^{(i)})$**.
- **$\text{Recency}(e_j) = \exp\left( -\lambda \cdot (t - t_j) \right)$**: Hàm suy giảm lũy thừa thời gian theo nguyên lý ACT-R.
- **$\text{Repetition}(e_j) = \frac{\text{count}(\text{pattern}(e_j))}{|W|}$**: Tần suất lặp lại của mô thức hành vi.

**Điều kiện Kích hoạt Phản tư**:
$$\text{Evidence\_Normalized}_t(i) \ge \theta_P \quad (\text{với } \theta_P \in [0.0, 1.0])$$

**Ngưỡng Thích ứng**:
$$\theta_P = \min\left(0.95, \theta_{\text{base}} \cdot \left( 1 + \alpha \cdot \text{Stability}(P_t^{(i)}) \right)\right)$$
- $\theta_{\text{base}} = 0.60$, $\alpha = 0.30$.
- $\text{Stability}(P_t) = \text{mean}(\text{BigFive}) \cdot (1.0 - \text{Neuroticism})$.
- **Ý nghĩa**: Dù NPC trải qua 10.000 sự kiện vặt, điểm $\text{Evidence\_Normalized}$ luôn bị chặn trên tại 1.0, buộc nhân cách chỉ thay đổi khi có đủ mật độ biến cố đặc biệt nghiêm trọng.

---

### 2.3. Cơ chế Ép buộc Cấu trúc & Phục hồi (Constrained Decoding & Safe-Fail Fallback)

Để biến hộp đen LLM thành hộp trắng lập trình được, hệ thống loại bỏ hoàn toàn prompt tự do:
1. **Constrained Decoding**: Sử dụng schema JSON ép buộc (thông qua Pydantic / JSON-mode của vLLM/Ollama) yêu cầu mô hình xuất chính xác định dạng:
   ```json
   {
     "delta_valence": float,
     "delta_arousal": float,
     "delta_anger": float,
     "confidence": float,
     "reasoning": string
   }
   ```
2. **Safe-Fail Fallback**:
   - Nếu parser thất bại (LLM sinh text thừa), hệ thống retry tối đa 2 lần với prompt sửa lỗi.
   - Nếu vẫn thất bại: Ghi log cảnh báo và gán $\Delta E = 0$ (**Safe-Fail**) để không làm hỏng state.
   - Nếu $\text{confidence} < 0.60$: Tự động fallback về $\Delta E = 0$ để tránh ảo giác.
   - Nếu $\text{confidence} \ge 0.60$: Kẹp biên độ nghiêm ngặt: $|\Delta_{\text{validated}}| = \min(|\Delta_{\text{raw}}|, \Delta_{\max}(\text{event}))$.

---

### 2.4. Cơ chế Phân xử Xung đột giữa các Module (Conflict Resolution & Priority Arbitration)

Khi các module đưa ra các tín hiệu trái ngược nhau (ví dụ: Ký ức nhớ đối phương từng giúp đỡ $M > 0$, nhưng cảm xúc hiện tại rất sợ hãi $\text{Fear} > 0.8$), hệ thống áp dụng **Cây phân cấp ưu tiên (Priority Hierarchy)**:

$$\text{Priority}: \quad \text{SAFETY (1.0)} > \text{IDENTITY (0.9)} > \text{RELATIONSHIP (0.7)} > \text{EMOTION (0.5)} > \text{MEMORY (0.3)}$$

**Hàm lựa chọn hành vi**:
$$\text{Action}_t = \arg\max_{a} \sum_{k \in \{S, I, R, E, M\}} \text{Weight}_k \cdot \text{Utility}(a, \text{State}_k)$$

---

## 3. GIẢI QUYẾT 3 ĐIỂM MÙ KỸ THUẬT (RESOLVING 3 TECHNICAL BLIND SPOTS)

### 🕳️ Điểm mù 1: Quản lý Tràn Context Window trên Laptop 8B bằng RAG
- **Vấn đề**: Khi NPC tích lũy 500 sự kiện, làm sao nhét hết vào context window (8k–32k) để Deep Appraisal và Reflection?
- **Giải pháp**: Hệ thống không nhồi toàn bộ lịch sử thô. Chúng tôi ứng dụng **RAG kết hợp RecMem**:
  - Khi có biến cố $e_t$, hệ thống dùng Vector Embedding để truy xuất **Top-5 ký ức liên quan nhất** + **3 ký ức gần nhất (Recency)** để nạp vào Context Window của LLM 8B.
  - Tầng đệm tiềm thức RecMem đóng vai trò là bộ lọc: Chỉ những sự kiện lặp lại hoặc có cảm xúc mạnh mới được củng cố (consolidate) đưa vào Vector DB, giữ Context Window luôn dưới 2048 tokens.

### 🕳️ Điểm mù 2: Mạng Wi-Fi Chập chờn & Lệch State (Optimistic UI & Reconciliation)
- **Vấn đề**: Đang tương tác mà rớt mạng Wi-Fi, Phone không nhận được lệnh cập nhật state từ Laptop, NPC bị lệch pha tâm lý.
- **Giải pháp**:
  - **Optimistic UI**: Phone tự động cập nhật cảm xúc vi mô cục bộ (Fast State) dựa trên SLM để cuộc đối thoại không bị ngắt mạch.
  - **Reconciliation Engine**: Khi có kết nối trở lại, Phone gửi một gói `state_delta` lên Laptop. Laptop đóng vai trò là **Source of Truth (Nguồn chân lý duy nhất)**, dùng logic Evidence Gate để thẩm định lại. Nếu hợp lệ thì merge vào Global State, nếu xung đột thì Laptop gửi lệnh rollback về Phone.

### 🕳️ Điểm mù 3: Bộ Dữ liệu Chuẩn hóa (Ground-Truth Affective Dataset)
- **Vấn đề**: Không thể tự thuê chuyên gia tâm lý gán nhãn hàng trăm câu thoại sandbox từ con số 0.
- **Giải pháp**:
  - Tận dụng 2 tập benchmark học thuật công khai đã được chuẩn hóa: **GoEmotions** (để validate module Surface Emotion trên Phone) và **MELD** (để validate module Deep Appraisal trên Laptop).
  - Đối với kịch bản Sandbox, sử dụng phương pháp **LLM-as-a-Judge (GPT-4o)** làm điểm tham chiếu chuẩn (reference point) và **đối chiếu chéo (cross-validate) với 3 annotator là người thật trên 50 mẫu đại diện** (đo chỉ số Cohen's Kappa $\kappa$).

---

## 4. GIAO THỨC TRUYỀN LUỒNG & CHE GIẤU ĐỘ TRỄ (STREAMING & UX MASKING)

- **Token Streaming qua WebSocket**: Áp dụng Server-Sent Events (SSE). Mục tiêu **Time-to-First-Token (TTFT) < 400ms**, chữ cái đầu tiên hiển thị lên màn hình gần như tức thì.
- **UX Masking (Che giấu tâm lý)**: Trong khoảng thời gian Laptop 8B xử lý Deep Appraisal (0.8 – 1.5s), Phone Worker sẽ tự động kích hoạt **Animation Trạng thái** (Idle animation, nhíu mày, nhìn xa, hiệu ứng typing *"Đang suy nghĩ..."*). Điều này biến độ trễ kỹ thuật thành độ trễ "nhập vai" tự nhiên của con người.

---

## 5. MA TRẬN ĐÁNH GIÁ THỰC NGHIỆM: BASELINES & ABLATIONS

| Điều kiện kiểm thử | Cấu hình kỹ thuật | Mục đích kiểm chứng khoa học |
| :--- | :--- | :--- |
| **Baseline 1: Flat LLM** | Prompt nhập vai thông thường, KHÔNG Memory, KHÔNG Evolution | Chứng minh sự cần thiết của hệ thống quản lý trạng thái. |
| **Baseline 2: Scripted FSM** | Cây hội thoại cứng + Máy trạng thái hữu hạn truyền thống | Đo lường tính linh hoạt và độ tự nhiên so với phương pháp game cũ. |
| **Baseline 3: Un-gated LLM** | LLM cập nhật tính cách sau MỌI sự kiện (không có Gate) | **Chứng minh Evidence Gate ngăn chặn hiện tượng Trôi nhân vật (Zero-drift).** |
| **Ablation A: w/o Memory** | Giữ Evidence Gate nhưng ngắt kết nối Episodic Memory | Cô lập đóng góp của ký ức tích lũy đối với sự phát triển nhân vật. |
| **Ablation B: w/o Appraisal**| Gán nhãn cảm xúc bề mặt ngẫu nhiên, bỏ Deep Appraisal | Cô lập vai trò của thẩm định nhận thức đối với độ chân thực phản ứng. |
| **Ablation C: w/o Conflict** | Bỏ cơ chế Cây ưu tiên phân xử xung đột (Priority Hierarchy) | Đo lường tỷ lệ sinh hành vi phi lý khi cảm xúc mâu thuẫn ký ức. |
| **Proposed System** | **Đầy đủ 5 Module + Evidence Gate + Priority Resolution** | Chứng minh đạt sự cân bằng tối ưu giữa tính linh hoạt và tính ổn định. |

---

## 6. PHÂN TÍCH TẢI TÍNH TOÁN & ĐỘ TRỄ HỆ THỐNG (COMPLEXITY ANALYSIS)

| Tác vụ kỹ thuật | Ước tính FLOPs | Chiếm dụng RAM | Độ trễ dự kiến | Thiết bị thực thi |
| :--- | :--- | :--- | :--- | :--- |
| Surface Emotion Detection | ~0.5 GFLOPs | ~150 MB | 50 – 80 ms | Phone CPU |
| Dialogue Generation (0.5B, 50 tok) | ~15 GFLOPs | ~1.2 GB | 2.0 – 3.5 s (TTFT < 400ms) | Phone NPU/CPU |
| Deep Cognitive Appraisal (8B, 1 turn)| ~200 GFLOPs | ~5.8 GB VRAM | 0.8 – 1.5 s (Che bởi UX Masking) | Laptop GPU RTX 4060 |
| Evidence Gate & ACT-R Computation | ~0.01 GFLOPs | ~15 MB | < 2 ms | Laptop CPU |
| Deep Reflection & Trait Evolution | ~500 GFLOPs | ~5.8 GB VRAM | 4.0 – 7.0 s (Chạy định kỳ) | Laptop GPU RTX 4060 |
| WebSocket LAN Relay | N/A | ~10 MB | 5 – 25 ms | Mạng Wi-Fi nội bộ |

---

## 7. LỘ TRÌNH 8 TUẦN ĐIỀU CHỈNH CHIẾN LƯỢC PHONE FARM

Áp dụng chiến thuật *"Lùi 1 bước để tiến 3 bước"*, đảm bảo an toàn tuyệt đối về mặt kỹ thuật:

- **Phase 1 (Tuần 1–2): Formal State Engine & Schema Chuẩn**
  - Xây dựng toán học `formal_state.py`, kiểm thử bão hòa $\tanh$, cơ chế Fallback Safe-fail và Conflict Priority. *(ĐÃ HOÀN THÀNH)*.
- **Phase 2 (Tuần 3–4): Backend Thống nhất & MVP Giả lập trên Laptop** *(ĐÃ HOÀN THÀNH & ĐÁNH GIÁ)*
  - Xây dựng FastAPI Backend trên Laptop mô phỏng hoàn chỉnh luồng: Laptop (LLM 8B Deep Appraisal) giao tiếp với Phone Simulator (giả lập 2 NPC Alice & Bob qua WebSocket).
  - Khảo sát thực nghiệm hiện tượng *Semantic Collapse* trên SLM 0.5B và nâng cấp thành công lên **`Qwen2.5-1.5B-Instruct`** (đạt chất lượng Surface Text mạch lạc, loại bỏ hoàn toàn ảo giác y tế và câu từ lặp vụng về, tốc độ 20 t/s, khả thi INT4 ~1.0GB RAM trên Android).
  - Đạt 15/15 unit/integration tests tại `tests/test_phase2_mvp.py` và demo thời gian thực tại `sandbox/run_phase2_demo.py`.
- **Phase 3 (Tuần 5–6): RAG Memory Buffer & Reflection Engine** *(ĐÃ HOÀN THÀNH & KIỂM CHỨNG TOÀN DIỆN)*
  - Xây dựng bộ nhớ sự kiện (Episodic Memory Buffer) dạng mock JSON nhẹ (`sandbox/memories_{agent_id}.json`) không phụ thuộc server DB nặng.
  - Tích hợp hàm suy giảm trí nhớ sinh học **ACT-R Decay**: $A_i(t) = \ln \sum (t - t_k + \epsilon)^{-d} + 0.15 \ln(1 + \text{recall\_count})$ và cơ chế củng cố RecMem cho các sự kiện được gợi nhớ nhiều lần ($A_i > 1.43$ vs unconsolidated $< 0.58$).
  - Cửa sổ ngữ cảnh giới hạn chặt chẽ (**Bounded RAG**): **Top-5 Semantic + Top-3 Recency** (chặn cứng $\le 8$ items $\le 450$ tokens), loại bỏ hoàn toàn nguy cơ tràn context window trên Edge SLM.
  - Triển khai **Deep Reflection Engine** trên Host: Khi Cổng chứng cứ bão hòa đạt ngưỡng ($\text{Evidence\_Normalized} \ge \theta_P$, đạt $0.7301 \ge 0.6772$ sau 3 biến cố thù địch liên tiếp), Host tổng hợp 3 High-Level Insights (Park et al., 2023) và kích hoạt tiến hóa vĩnh viễn Trạng thái Chậm (Slow State $P_t$): Niềm tin cốt lõi chuyển sang ranh giới tự vệ, Worldview Trust giảm $0.70 \to 0.50$, Agreeableness giảm $0.80 \to 0.74$.
  - Giải tỏa áp lực cổng bằng chứng về $0.0003$ để ngăn vòng lặp kích hoạt liên tục và tự động ghi bản ghi ký ức `[CHIÊM NGHIỆM SÂU SẮC]` vào JSON store.
  - Đạt **21/21 unit & integration tests (100% pass)** tại `tests/test_phase3_reflection.py` và demo thực địa trực quan 6 lượt tại `sandbox/run_phase3_demo.py` với SLM 1.5B trên GPU.
- **Phase 4 (Tuần 7–8): Đánh giá Đa điều kiện & Demo Trực quan Trên Máy** *(ĐÃ HOÀN THÀNH & KIỂM CHỨNG TOÀN DIỆN)*
  - Thực hiện đánh giá khoa học đa điều kiện tại [`sandbox/run_phase4_benchmark.py`](file:///home/zafkiel/Workspace/PhoneFarm/sandbox/run_phase4_benchmark.py), so sánh trực diện trên kịch bản áp lực 6 lượt giữa **Flat LLM**, **Un-gated LLM**, và **Hệ thống đề xuất (Proposed Saturated-Gated System)**.
  - Chứng minh định lượng bằng dữ liệu thực nghiệm lưu tại [`sandbox/phase4_benchmark_results.jsonl`](file:///home/zafkiel/Workspace/PhoneFarm/sandbox/phase4_benchmark_results.jsonl):
    * **Độ trôi nhân vật (Gaslighting Drift)**: Hệ thống đề xuất đạt **0.00 (Kháng trôi dạt hoàn hảo)** so với **0.55** của Flat LLM (mất trí nhớ) và **0.55** của Un-gated LLM (nhảy cóc tính cách hỗn loạn).
    * **Khả năng truy vết (Auditability)**: Đạt **100%** chuỗi vết chuyển trạng thái từ biến cố, thẩm định Scherer CPM đến bản ghi ký ức.
    * **Tính hai chiều của tiến hóa (Bidirectional Arc)**: Kiểm chứng thành công cả nhánh tiến hóa phòng vệ (Alice: vị tha $\to$ tự vệ) và nhánh tiến hóa hợp tác (Bob: cô lập $\to$ mở lòng hợp tác).
  - Triển khai **Giao diện Web Trực quan Tương tác Thời gian thực (Interactive Web Dashboard)** tại `GET /` ([`sandbox/web_dashboard.html`](file:///home/zafkiel/Workspace/PhoneFarm/sandbox/web_dashboard.html)): Tích hợp đồng hồ đo cảm xúc vi mô (Valence, Anger), thanh bão hòa Cổng bằng chứng $\tanh$, giám sát tiến hóa nhân cách cốt lõi, bộ duyệt ký ức RAG với điểm ACT-R và 6 nút kịch bản thao tác 1-chạm.
  - Đạt **25/25 unit & integration tests (100% pass)** tại [`tests/test_phase4_evolution.py`](file:///home/zafkiel/Workspace/PhoneFarm/tests/test_phase4_evolution.py).
