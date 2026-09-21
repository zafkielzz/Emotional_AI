# BÁO CÁO KẾT QUẢ BENCHMARK: MODULE 3 - EPISODIC MEMORY ENGINE

- **Ngày kiểm thử**: 2026-09-11 19:37:02
- **Môi trường thực thi**: Conda `capstone` (PyTorch, Transformers)
- **Mô hình nhúng**: `sentence-transformers/all-MiniLM-L6-v2` (384-d normalized)
- **Cơ sở dữ liệu**: SQLite embedded (`data/benchmark_module3_memory.sqlite3`)
- **Kết quả chung**: **8/8 bài kiểm tra ĐẠT (100.0% Pass Rate)**
- **Tổng thời gian**: 3.20 giây

## 1. BẢNG TỔNG HỢP ĐỐI SÁNH KHOA HỌC

| STT | Tên Bài Kiểm Thử | Cơ Sở Lý Thuyết / Thuật Toán | Kết Quả Đo Kiểm | Trạng Thái |
| :--- | :--- | :--- | :--- | :---: |
| 1 | **Suy giảm lũy thừa thời gian** | ACT-R Power-Law Decay | 1m > 1h > 1d > 7d (Đơn điệu giảm) | ✅ PASS |
| 2 | **Ký ức đèn Flash (Kháng suy giảm)** | Amygdala Flashbulb Modulation | Trauma: -1.96 vs Minor: -6.86 ($\Delta = +4.90$) | ✅ PASS |
| 3 | **Củng cố qua luyện tập (RecMem)** | Power Law of Practice & Spaced Repetition | Practiced: -2.42 vs Unrecalled: -4.51 ($\Delta = +2.09$) | ✅ PASS |
| 4 | **Bóc tách ngữ nghĩa dày đặc** | 384-d Dense Cosine Disentanglement | Top-1 Medical (0.46) & Top-1 Threat (0.46) | ✅ PASS |
| 5 | **Ràng buộc cửa sổ ngữ cảnh** | Working Memory Budget ($\le 8$ items, $\le 450$ tok) | Retrieved: 5 items (~238 tokens) | ✅ PASS |
| 6 | **Cô lập trí nhớ đa nhân vật** | Agent Partitioning (Aiden $\cap$ Lyra $= \emptyset$) | Zero Bleed confirmed giữa Aiden và Lyra | ✅ PASS |
| 7 | **Độ trễ truy xuất trên Laptop** | Sub-millisecond Matrix Multiplications | Mean: 7.37 ms (Target < 15ms) | ✅ PASS |
| 8 | **Áp lực Cây kim trong đáy bể (N=220)** | Long-term Needle-in-a-Haystack across 30 days | **5/5 Needles Top-1 Exact (100.0%)** | ✅ PASS |
