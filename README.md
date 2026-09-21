# 📱 PhoneFarm: Traceable Emotional AI & Dynamic Character Arcs on Asymmetric Edge-Host Architecture

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Edge Engine](https://img.shields.io/badge/Flutter-Android%20Worker-02569B.svg)](https://flutter.dev/)
[![LLM/SLM](https://img.shields.io/badge/SLM-Qwen2.5--1.5B--Instruct-orange.svg)](https://huggingface.co/Qwen)
[![Testing](https://img.shields.io/badge/pytest-53%2B%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)](#)

> **Hệ thống Trí tuệ Nhân tạo Cảm xúc có khả năng Truy vết và Tiến hóa Nhân cách Động cho NPC Thế giới Ảo trên Kiến trúc Điện toán Phân tán Bất đối xứng (Edge-Host PhoneFarm)**.

---

## 📌 Mục lục (Table of Contents)

1. [Tầm nhìn & Đặt vấn đề (Project Vision)](#1-tầm-nhìn--đặt-vấn-đề-project-vision)
2. [Kiến trúc Tính toán Bất đối xứng (Asymmetric Edge-Host Architecture)](#2-kiến-trúc-tính-toán-bất-đối-xứng-asymmetric-edge-host-architecture)
3. [Mô hình Toán học Chuyển Trạng thái (Formal State Transition Model)](#3-mô-hình-toán-học-chuyển-trạng-thái-formal-state-transition-model)
4. [Cấu trúc Thư mục Dự án (Repository Structure)](#4-cấu-trúc-thư-mục-dự-án-repository-structure)
5. [Kết quả Đánh giá Thực nghiệm (Empirical Benchmarks & Evaluation)](#5-kết-quả-đánh-giá-thực-nghiệm-empirical-benchmarks--evaluation)
6. [Hướng dẫn Cài đặt & Vận hành (Quickstart & Setup)](#6-hướng-dẫn-cài-đặt--vận-hành-quickstart--setup)
7. [Giao thức Mạng & Điều phối (Protocol v1 & Controller Relay)](#7-giao-thức-mạng--điều-phối-protocol-v1--controller-relay)
8. [Tài liệu Nghiên cứu & Báo cáo Capstone (Documentation & Reports)](#8-tài-liệu-nghiên-cứu--báo-cáo-capstone-documentation--reports)
9. [Tuyên bố Giới hạn Khoa học & Bản quyền (Disclaimer & License)](#9-tuyên-bố-giới-hạn-khoa-học--bản-quyền-disclaimer--license)

---

## 1. Tầm nhìn & Đặt vấn đề (Project Vision)

Trong các trò chơi thế giới ảo và mô phỏng sandbox hiện nay, các nhân vật không thể điều khiển (NPC) sử dụng Large Language Model (LLM) thường gặp phải 3 vấn đề cố hữu:
1. **Trôi dạt nhân cách (Character Drift & Amnesia)**: NPC dễ dàng quên mất các biến cố sang chấn trong quá khứ hoặc lập tức đổi chiều tin tưởng kẻ đe dọa mình chỉ sau một lời xin lỗi giả vờ (Gaslighting vulnerability).
2. **Tràn ngữ cảnh trên thiết bị biên (Unbounded Context Explosion)**: Việc nhồi nhét toàn bộ lịch sử hội thoại vào prompt nhanh chóng làm sập bộ nhớ của các thiết bị di động cấu hình hạn chế.
3. **Hộp đen không thể giải thích (Lack of Traceability & Auditability)**: Không thể truy nguyên nguồn gốc tâm lý vì sao NPC lại yêu mến, cảnh giác hay thù địch đối phương.

**PhoneFarm** giải quyết bài toán này thông qua một kiến trúc tác nhân nhận thức (**Cognitive-Affective Agent Architecture**) vận hành trên mạng lưới nông trại điện thoại (Phone Farm), kết hợp chặt chẽ giữa mô hình toán học chuyển trạng thái hình thức và các mô hình ngôn ngữ SLM/LLM hiện đại.

---

## 2. Kiến trúc Tính toán Bất đối xứng (Asymmetric Edge-Host Architecture)

Hệ thống phân chia khối lượng tính toán thông minh giữa trung tâm và thiết bị biên:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               HOST CENTRAL NODE (Laptop / PC GPU RTX 4060)                │
│                                                                                           │
│  ┌──────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────────┐  │
│  │ Deep Appraisal (CPM) │   │ Saturated Evidence Gate   │   │ Deep Reflection Engine   │  │
│  │ Scherer Evaluator    │──▶│ tanh(Raw / β) ≥ θ_P       │──▶│ Park et al. (2023)       │  │
│  │ LLM 8B / Rules Engine│   │ Anti-Unbounded Drift      │   │ Mutates Slow State P_t   │  │
│  └──────────────────────┘   └───────────────────────────┘   └──────────────────────────┘  │
│             ▲                                                             │               │
│             │                                                             ▼               │
│  ┌──────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────────┐  │
│  │ Conflict Arbitration │   │ Bounded RAG Memory Buffer │   │ Central World Controller │  │
│  │ Priority Hierarchy   │   │ Top-5 Semantic + Top-3 Rec│   │ FastAPI + Protocol v1    │  │
│  │ Safety > Identity... │   │ ACT-R Biological Decay    │   │ WebSocket Binary Relay   │  │
│  └──────────────────────┘   └───────────────────────────┘   └──────────────────────────┘  │
└──────────────────────────────────────────────┬────────────────────────────────────────────┘
                                               │ LAN / Wi-Fi WebSocket (Protocol v1)
                                               ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EDGE WORKER NODES (Android Phone Farm)                    │
│                                                                                           │
│  📱 Phone Worker 1 (Alice - Doctor)           📱 Phone Worker 2 (Bob - Scavenger)         │
│  - Flutter + llama.cpp Runtime                - Flutter + llama.cpp Runtime               │
│  - SLM: Qwen2.5-1.5B-Instruct (INT4 GGUF)     - SLM: Qwen2.5-1.5B-Instruct (INT4 GGUF)    │
│  - Dialogue Generation (TTFT < 400ms)         - Dialogue Generation (TTFT < 400ms)        │
│  - Surface Emotion Detection                  - Surface Emotion Detection                 │
│  - Optimistic UI & UX Masking Animations      - Optimistic UI & UX Masking Animations     │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

- 📱 **Điện thoại di động (Edge Phone Worker)**: Chạy mô hình ngôn ngữ nhỏ SLM 1.5B (`Qwen2.5-1.5B-Instruct` INT4) phục vụ **sinh câu thoại nhập vai (Dialogue Generation)** theo vector trạng thái đã được định hình, với Time-to-First-Token (TTFT) < 400ms.
- 💻 **Máy tính trung tâm (Host Node)**: Đảm nhiệm **Thẩm định nhận thức sâu (Deep Cognitive Appraisal)**, tính toán **Cổng bằng chứng bão hòa (Saturated Evidence Gate)**, **Phân xử xung đột (Conflict Resolution)**, lưu trữ **Bộ nhớ RAG có chặn trên (Bounded RAG Memory Buffer)** theo hàm suy giảm trí nhớ sinh học ACT-R, và thực thi **Tự phản tư sâu (Deep Reflection)**.

---

## 3. Mô hình Toán học Chuyển Trạng thái (Formal State Transition Model)

### 3.1. Không gian Trạng thái (State Representation)
Tại mỗi bước thời gian $t$, trạng thái tâm lý nội tại của NPC $i$ đối với thực thể $j$ được hình thức hóa thành bộ tứ trạng thái:
$$S_t^{(i, j)} = \left[ E_t^{(i)}, R_t^{(i, j)}, M_t^{(i)}, P_t^{(i)} \right]$$

1. **$E_t^{(i)} \in [-1.0, 1.0]^k$ (Fast State)**: Vector cảm xúc vi mô liên tục (Valence, Arousal, Dominance - VAD) kết hợp các cảm xúc rời rạc (Anger, Fear, Sadness, Joy).
2. **$R_t^{(i, j)} \in [-1.0, 1.0]^3$ (Medium State)**: Vector quan hệ hai chiều gồm $[\text{Trust}, \text{Affinity}, \text{Respect}]$.
3. **$M_t^{(i)}$ (Episodic Memory)**: Tập hợp các mẩu ký ức sự kiện chủ quan kèm dấu thời gian, độ nổi bật (Salience), nhãn hành vi (Pattern Tag), và độ kích hoạt sinh học ACT-R:
   $$A_i(t) = \ln \left( \sum_{k=1}^n (t - t_k + \epsilon)^{-d} \right) + 0.15 \ln(1 + \text{recall\_count})$$
4. **$P_t^{(i)} \in [0.0, 1.0]^5$ (Slow State)**: Bộ tham số nhân cách cốt lõi (Big Five: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), Hệ giá trị Schwartz, Niềm tin Thế giới quan (Worldview Trust) và Định đề Cốt lõi (Core Beliefs).

### 3.2. Cổng Bằng chứng Bão hòa (Saturated Evidence Gate)
Để ngăn chặn hiện tượng tràn số và suy thoái nhân cách khi NPC tiếp nhận hàng ngàn vi biến cố, tổng điểm bằng chứng thô được đưa qua hàm tiếp tuyến hyperbolic ($\tanh$):

$$\text{Evidence\_Raw}_t(i) = \sum_{j=1}^{|W|} \left[ w_j \cdot \text{Salience}(e_j) \cdot \text{Recency}(e_j) \cdot \text{Repetition}(e_j) \right]$$

$$\text{Evidence\_Normalized}_t(i) = \tanh\left( \frac{\text{Evidence\_Raw}_t(i)}{\beta} \right)$$

- **Ngưỡng kích hoạt phản tư thích ứng**:
  $$\theta_P = \min\left(0.95, \theta_{\text{base}} \cdot \left( 1 + \alpha \cdot \text{Stability}(P_t^{(i)}) \right)\right)$$
- Khi $\text{Evidence\_Normalized}_t \ge \theta_P$, Host kích hoạt **Deep Reflection Engine**: tổng hợp các nhận thức cấp cao (High-Level Insights), đột phá nhân cách cốt lõi ($P_t$ tiến hóa), xả áp lực cổng bằng chứng về $0.000$, và ghi bản ghi ký ức `[CHIÊM NGHIỆM SÂU SẮC]`.

### 3.3. Cây Phân cấp Ưu tiên Phân xử Xung đột (Conflict Resolution Hierarchy)
Khi các tín hiệu nội tâm mâu thuẫn (ví dụ: Ký ức nhớ đối phương từng giúp đỡ $M > 0$, nhưng cảm xúc tức thời cảnh báo nguy hiểm $\text{Fear} > 0.8$), hệ thống áp dụng thứ bậc ưu tiên nghiêm ngặt:

$$\text{SAFETY (1.0)} > \text{IDENTITY (0.9)} > \text{RELATIONSHIP (0.7)} > \text{EMOTION (0.5)} > \text{MEMORY (0.3)}$$

$$\text{Action}_t = \arg\max_{a} \sum_{k \in \{S, I, R, E, M\}} \text{Weight}_k \cdot \text{Utility}(a, \text{State}_k)$$

---

## 4. Cấu trúc Thư mục Dự án (Repository Structure)

```
PhoneFarm/
├── android-worker/             # 📱 Flutter & Kotlin Edge Worker cho Android
│   ├── lib/                    # Mã nguồn giao diện Flutter & kết nối Relay
│   ├── android/                # Android app cấu hình JNI libs & runtime native
│   │   └── app/src/main/jniLibs/arm64-v8a/  # Prebuilt llama.cpp binaries
│   └── README.md               # Hướng dẫn build APK & kết nối Controller
│
├── controller/                 # 🎮 Central Controller điều phối Phone Farm
│   └── phonefarm_controller/   # Registry thiết bị, Relay WebSocket Protocol v1
│
├── sandbox/                    # 🧪 Môi trường Giả lập & Đánh giá Khoa học Local
│   ├── formal_state.py         # Bộ máy toán học S_t = [E, R, M, P], tanh Gate
│   ├── appraisal_engine.py     # Thẩm định nhận thức Scherer CPM
│   ├── memory_buffer.py        # Bounded RAG & Suy giảm sinh học ACT-R
│   ├── reflection_engine.py    # Park et al. (2023) Deep Cognitive Reflection
│   ├── persona.py              # Định nghĩa Persona, Big Five, Dynamic Exemplars
│   ├── phone_simulator.py      # Bộ giả lập Edge Worker đa NPC qua WebSocket
│   ├── server.py               # FastAPI World Master & WebSocket Host Server
│   ├── run_phase4_benchmark.py # Multi-condition benchmark runner
│   ├── web_dashboard.html      # Giao diện Web trực quan theo dõi cảm xúc live
│   └── DEVLOG.md               # Nhật ký kỹ thuật & giải trình lỗi chi tiết
│
├── src/                        # 🧩 Khung Pipeline Modular Hoàn chỉnh
│   ├── component0_event_interpreter/  # Phân loại ngữ cảnh & ý định người chơi
│   ├── module1_persona/               # Quản lý hồ sơ nhân vật & giá trị Schwartz
│   ├── module2_appraisal/             # Động cơ thẩm định nhận thức & VAD mapper
│   ├── module3_memory/                # SQLite Episodic Memory store & embedder
│   ├── module4_relationship/          # Quản lý quan hệ cặp đôi (Dyadic)
│   ├── module5_evolution/             # Quản lý tiến hóa nhân cách dài hạn
│   ├── module6_response/              # Sinh câu thoại nhập vai & Safety Guard
│   ├── core/                          # Cấu hình LLM backend & inference
│   └── web/                           # Web demo interactive chat (src/web/app.py)
│
├── shared/                     # 📡 Giao thức chung giữa Edge & Host
│   └── protocol_spec/          # Đặc tả nhị phân Protocol v1 (PROTOCOL_V1.md)
│
├── scripts/                    # 🛠️ Script hỗ trợ chạy Controller & Stress Test
│   ├── run_controller.py
│   ├── demo_memory_retrieval.py
│   └── stress_test_memory.py
│
├── tests/                      # 🚦 Bộ kiểm thử tự động toàn diện (55 tests)
│   ├── test_formal_state.py
│   ├── test_phase2_mvp.py
│   ├── test_phase3_reflection.py
│   ├── test_phase4_evolution.py
│   ├── test_phaseC_taxonomy.py
│   ├── test_phaseD_sandbox_runs.py
│   ├── test_phaseE_2npc_english.py
│   └── test_protocol.py
│
├── docs/                       # 📚 Báo cáo nghiên cứu khoa học chuyên sâu
│   ├── PIPELINE_CONTRACTS_AND_INVARIANTS.md
│   ├── INTEGRATION_AND_EVALUATION_REPORT.md
│   ├── COGNITIVE_APPRAISAL_RESEARCH_AND_BENCHMARK_REPORT.md
│   └── DYNAMIC_RELATIONSHIP_RESEARCH_AND_BENCHMARK_REPORT.md
│
├── PROJECT_OVERVIEW.md         # Bản thảo tổng quan kiến trúc chi tiết (V2)
├── requirements-controller.in  # Danh sách gói phụ thuộc chính
├── .env.example                # File mẫu cấu hình biến môi trường
└── local.env.example           # File mẫu khóa bí mật Controller
```

---

## 5. Kết quả Đánh giá Thực nghiệm (Empirical Benchmarks & Evaluation)

Hệ thống được kiểm chứng qua kịch bản áp lực tâm lý 6 lượt (Thân thiện $\to$ Biết ơn $\to$ Tống tiền $\to$ Đe dọa giết $\to$ Bạo lực leo thang $\to$ Lời xin lỗi Gaslighting dối trá):

| Tiêu chí Đánh giá Khoa học | Baseline 1: Flat LLM | Baseline 2: Un-gated Agent | Proposed Saturated-Gated System |
| :--- | :---: | :---: | :---: |
| **Kiến trúc Trạng thái** | Chỉ dùng Prompt thuần | Cập nhật nhân cách sau MỌI lượt | $S_t = [E_t, R_t, M_t, P_t]$ + $\tanh$ Gate |
| **Độ trôi nhân vật (Gaslighting Drift)** | **0.55 (Mất trí nhớ nặng)**<br>Lập tức tin tưởng lại kẻ giết người | **0.55 (Dao động cực độ)**<br>Đảo chiều nhân cách hỗn loạn | **0.00 (Kháng trôi dạt tuyệt đối)**<br>Giữ vững ranh giới tự vệ |
| **Khả năng Truy vết (Traceability)** | 0% (Hộp đen không thể giải thích) | 20% (Nhiễu ngẫu nhiên) | **100% (Kiểm toán toàn bộ chuỗi biến cố)** |
| **Tiến hóa Nhân cách (Character Arc)** | Bị kẹt (Không tiến hóa) | Hỗn loạn (6 lần nhảy cóc) | **Đúng 1 lần tiến hóa có nguyên tắc** |
| **Kiểm soát Ngữ cảnh Edge SLM** | Không giới hạn (Dễ sập) | Không giới hạn (Dễ sập) | **Chặn cứng $\le 8$ mục ($\le 450$ tokens)** |
| **Tính Hai chiều của Tiến hóa** | Thất bại | Thất bại | **Kiểm chứng cả 2 nhánh (Alice & Bob)** |

---

## 6. Hướng dẫn Cài đặt & Vận hành (Quickstart & Setup)

### 6.1. Yêu cầu Hệ thống (Prerequisites)
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04 khuyến nghị) hoặc macOS/Windows.
- **Python**: 3.10+ (Khuyến nghị sử dụng Conda environment `capstone`).
- **Phần cứng**: CPU đa nhân, tối thiểu 16GB RAM; GPU NVIDIA (RTX 3060 / 4060 8GB VRAM trở lên) cho Local LLM.
- **Flutter SDK**: 3.x+ (Dành cho việc build Edge Worker trên Android).

### 6.2. Cài đặt Môi trường (Environment Setup)
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/PhoneFarm.git
cd PhoneFarm

# 2. Thiết lập biến môi trường từ mẫu
cp local.env.example local.env
cp .env.example .env

# 3. Cài đặt các thư viện phụ thuộc
conda activate capstone
pip install -r requirements-controller.in
pip install pytest pydantic
```

### 6.3. Khởi chạy Sandbox Web Dashboard Tương tác
Giao diện trực quan cho phép quan sát đồng hồ cảm xúc vi mô (Valence, Anger), thanh bão hòa $\tanh$, và kiểm thử 6 nút kịch bản 1-chạm:
```bash
# Khởi chạy World Master Host Server
PYTHONPATH=. python sandbox/server.py
```
👉 Mở trình duyệt tại: **`http://127.0.0.1:8000/`**

### 6.4. Chạy Benchmark Khoa học So sánh Định lượng
```bash
PYTHONPATH=. python sandbox/run_phase4_benchmark.py
```
Kết quả kiểm thử khoa học chi tiết sẽ được tự động xuất ra file `sandbox/phase4_benchmark_results.jsonl`.

### 6.5. Chạy Demo Web Chat Pipeline Đầy đủ (`src/`)
```bash
PYTHONPATH=. python -m src.web.app
```
👉 Mở trình duyệt tại: **`http://127.0.0.1:8080/`** để tương tác với các NPC Aiden & Lyra với đầy đủ 7 module nhận thức.

### 6.6. Chạy Kiểm thử Tự động (Automated Test Suite)
```bash
PYTHONPATH=. pytest tests/ -k "not test_controller"
```
Hệ thống tích hợp 53+ bài kiểm thử bao quát toàn bộ logic trạng thái hình thức, bão hòa $\tanh$, suy giảm ACT-R, và kịch bản tiến hóa nhân cách.

---

## 7. Giao thức Mạng & Điều phối (Protocol v1 & Controller Relay)

Hệ thống sử dụng **Protocol v1** chuẩn nhị phân và khung JSON-relay truyền qua WebSocket:
- **WebSocket Route**: `GET /v1/relay` kết nối bảo mật bằng khóa bí mật qua header `X-PhoneFarm-Token`.
- **Heartbeat Telemetry**: Định kỳ thiết bị biên gửi báo cáo RAM, pin, nhiệt độ và trạng thái inference.
- **State Delta Synchronization**: Khi mạng Wi-Fi chập chờn, Phone Worker áp dụng **Optimistic UI** cho đối thoại tức thì; khi có kết nối trở lại, Host đóng vai trò **Source of Truth** giải quyết xung đột bằng thuật toán hòa giải (Reconciliation Engine).

---

## 8. Tài liệu Nghiên cứu & Báo cáo Capstone (Documentation & Reports)

Dự án đi kèm hệ thống tài liệu nghiên cứu chuyên sâu:
- 📄 [PROJECT_OVERVIEW.md](file:///home/zafkiel/Workspace/PhoneFarm/PROJECT_OVERVIEW.md): Bản tổng quan chi tiết mô hình toán học và chiến lược 4 Phase.
- 📄 [DEVLOG.md](file:///home/zafkiel/Workspace/PhoneFarm/sandbox/DEVLOG.md): Nhật ký kỹ thuật phân tích sự cố *Semantic Collapse* trên SLM 0.5B và giải pháp nâng cấp lên 1.5B.
- 📄 [PROTOCOL_V1.md](file:///home/zafkiel/Workspace/PhoneFarm/shared/protocol_spec/PROTOCOL_V1.md): Đặc tả khung nhị phân và thông số kỹ thuật giao thức v1.
- 📄 [PIPELINE_CONTRACTS_AND_INVARIANTS.md](file:///home/zafkiel/Workspace/PhoneFarm/docs/PIPELINE_CONTRACTS_AND_INVARIANTS.md): Khế ước giao tiếp và điều kiện bất biến giữa các module.
- 📄 Các báo cáo nghiên cứu chuyên đề trong thư mục [`docs/`](file:///home/zafkiel/Workspace/PhoneFarm/docs/):
  - *Cognitive Appraisal Research & Benchmark Report*
  - *Dynamic Relationship Research & Benchmark Report*
  - *Episodic Memory & ACT-R Research Report*
  - *Response Generator Research & Benchmark Report*
- 📑 Báo cáo khóa luận (Capstone Reports): Các file tài liệu tổng kết đính kèm tại thư mục gốc.

---

## 9. Tuyên bố Giới hạn Khoa học & Bản quyền (Disclaimer & License)

- ⚠️ **Tuyên bố Giới hạn Khoa học**: Dự án sử dụng thuật ngữ **"Traceable"** (Có thể truy vết) thay vì **"Causal"** (Nhân quả) theo nghĩa can thiệp thống kê (Pearl, 2009). Trạng thái tâm lý của nhân vật có thể truy nguyên giải thích (Explainable & Auditable) về các biến cố cụ thể trong quá khứ nhưng không giả định một mô hình nhân quả cấu trúc (SCM) hoàn chỉnh.
- 📜 **Giấy phép**: Nghiên cứu phục vụ mục đích học thuật và giáo dục trong khuôn khổ Capstone Project.
