# SIMC_2026 — Anomaly Detection on Smart Home Energy Data

Đồ án môn **DAP391m** (dự án backup). Phát hiện bất thường trong tiêu thụ điện
của nhà thông minh, dựa trên dataset **HomeC (Smart Home Dataset with Weather
Information)**. Có thể chỉnh để nộp hội nghị **SIMC 2026** (mục tiêu phụ).

---

## 1. Tổng quan

- **Bài toán:** phát hiện các khoảng thời gian tiêu thụ điện bất thường.
- **Dataset:** HomeC — 503.910 dòng × 32 cột, đo mỗi phút (01–12/2016), gồm
  điện năng theo thiết bị + dữ liệu thời tiết.
- **Hướng tiếp cận:** dataset KHÔNG có nhãn bất thường sẵn → tạo **proxy label**
  bằng ngưỡng thống kê (`use [kW] > mean + 3·std`, ~2.46% là bất thường) → đưa
  về bài toán **classification** để dùng được F1/AUC và so sánh nhiều model.
  > ⚠️ Đây là proxy label, KHÔNG phải ground-truth thật. Phải nêu rõ giới hạn
  > này khi báo cáo và trả lời giảng viên.

---

## 2. Cấu trúc thư mục

```
SIMC_2026/
├── data/
│   └── HomeC_cleaned.csv          # dữ liệu đã làm sạch (đặt file ở đây)
├── notebooks/
│   └── A_model_pipeline.ipynb     # phần A: tạo nhãn + train model (Step 5-9)
├── src/
│   └── A_model_pipeline.py        # bản .py tương đương (chạy bằng terminal)
├── app/
│   ├── best_model.pkl             # model tốt nhất (notebook xuất ra)
│   ├── feature_columns.pkl        # thứ tự cột feature cho app
│   └── ... (dashboard của C)
├── report/
│   └── Day2_MemberB_Report.md     # phần B: problem, EDA, data understanding
└── README.md
```

---

## 3. Cài đặt

Yêu cầu Python 3.9+.

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm joblib
# cho dashboard (phần C):
pip install plotly dash
```

---

## 4. Cách chạy phần model (Member A)

### Notebook (khuyến nghị)
Mở `notebooks/A_model_pipeline.ipynb` trên Jupyter/Colab, chạy lần lượt từng cell
từ trên xuống.

### Hoặc chạy bằng terminal
```bash
python src/A_model_pipeline.py
```

Sau khi chạy xong sẽ sinh ra: `best_model.pkl`, `feature_columns.pkl`,
`model_comparison.csv`.

### ⚙️ DEV_MODE — đọc kỹ phần này
Trong cả notebook lẫn file `.py` có biến `DEV_MODE` ở đầu (SECTION 1):

| Giá trị | Ý nghĩa | Khi nào dùng |
| --- | --- | --- |
| `DEV_MODE = True`  | Chỉ lấy mẫu 100.000 dòng → chạy **nhanh** | Khi đang viết/sửa code, test thử cho lẹ |
| `DEV_MODE = False` | Dùng **toàn bộ ~500.000 dòng** | Khi train bản cuối để lưu model nộp bài |

**Quy trình đúng:** để `True` trong lúc dev cho nhanh → khi mọi thứ ổn, đổi sang
`False` và chạy lại 1 lần cuối để lấy model + số liệu chính thức.

> ⚠️ **Lưu ý tải máy:** giữ nguyên 500k dòng nên KHÔNG dùng SVM/One-Class SVM
> (rất chậm, dễ treo). Các model trong pipeline (LogReg, RandomForest, XGBoost,
> LightGBM, IsolationForest) đều chịu được khối lượng này.

> ⚠️ **Không thêm lại cột `use [kW]` vào feature.** Nhãn sinh ra từ chính cột
> này; nếu giữ nó làm feature sẽ bị data leakage (F1 giả ≈ 1.0).

---

## 5. Phân công

| Thành viên | Vai trò | Nội dung |
| --- | --- | --- |
| **A** | Model Lead | Tạo nhãn, Step 5–9 (split, train ≥5 model, tuning, pipeline) |
| **B** | Data & EDA | Step 1–3 (problem, RQ, metric, data understanding, EDA), báo cáo |
| **C** | Feature & Dashboard | Step 4 (feature engineering), app/dashboard |

Chi tiết theo ngày: xem file phân công công việc của nhóm.

---

## 6. Kết quả sơ bộ (mẫu dev, DEV_MODE=True)

| Model | F1 | AUC |
| --- | --- | --- |
| XGBoost | ~0.90 | ~0.998 |
| RandomForest | ~0.90 | ~0.998 |
| LightGBM | ~0.88 | ~0.998 |
| LogisticRegression | ~0.37 | ~0.965 |
| IsolationForest (unsupervised) | ~0.08 | ~0.727 |

> Con số trên là từ mẫu dev, sẽ cập nhật lại sau khi chạy full `DEV_MODE=False`.
> IsolationForest yếu hơn hẳn vì là unsupervised, không học từ proxy label —
> đây là điểm thảo luận tốt khi so sánh hai trường phái.

---

## 7. Ghi chú trung thực (đọc trước khi báo cáo/Q&A)

- F1 cao vì model học lại quy luật thống kê do chính nhóm đặt ra (proxy label),
  không phải phát hiện bất thường "thật" theo nghĩa nghiệp vụ.
- Khi giảng viên hỏi "nhãn lấy ở đâu ra" → trả lời thẳng: proxy label theo
  ngưỡng `mean + 3·std`, có trích baseline paper, và nêu rõ giới hạn.
- Phần "semantic" (cho SIMC) hiện chỉ đề cập nhẹ ở phần viết; đóng góp semantic
  còn mỏng so với yêu cầu của hội nghị.

---

## 8. Lưu ý pháp lý / học vụ

Mọi quyết định về điểm số, tính hợp lệ của dự án backup, hay lo ngại pháp lý của
dự án cũ nên xác nhận trực tiếp với **giảng viên hướng dẫn** và **bộ phận học vụ**.
