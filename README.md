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
│   └── HomeC_cleaned_final.rar    # dữ liệu đã làm sạch (CHƯA GIẢI NÉN)
├── notebooks/
│   ├── model_pipeline.ipynb       # phần A: tạo nhãn + train model (Step 5-9)
│   ├── best_model.pkl             # model tốt nhất (notebook xuất ra)
│   ├── feature_columns.pkl        # ⭐ thứ tự cột feature (APP CẦN FILE NÀY!)
│   └── model_comparison.csv       # bảng kết quả đánh giá model
├── src/
│   └── HomeC_preprocess.py        # phần B: tiền xử lý + EDA (Step 1-3)
├── app/
│   └── app.py                     # phần C: dashboard Plotly Dash + AI
├── report/
│   └── Day2_MemberB_Report.md     # phần B: problem, RQ, EDA, data understanding
├── visualization/
│   └── chart*.png                 # 4 biểu đồ (timeseries, histogram, boxplot, heatmap)
├── requirements.txt               # danh sách thư viện (dùng pip install -r)
└── README.md
```

---

## 3. Cài đặt

Yêu cầu Python 3.9+.

### Cách 1 — Cài từ requirements.txt (khuyến nghị)
```bash
pip install -r requirements.txt
```

### Cách 2 — Cài thủ công
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm joblib plotly dash requests jupyter
```

### Bước chuẩn bị dữ liệu
⚠️ **QUAN TRỌNG:** Dataset được nén dưới dạng `.rar` để tiết kiệm bộ nhớ. Trước khi chạy notebook hoặc app, bạn **PHẢI giải nén**:

1. Chuột phải vào `data/HomeC_cleaned_final.rar` 
2. Chọn "Extract Here" (cần WinRAR, 7-Zip, hoặc tương đương)
3. Sau giải nén, trong thư mục `data/` sẽ có file `HomeC_cleaned_final.csv` (~300MB)

Nếu chưa giải nén mà chạy notebook/app, sẽ báo lỗi `FileNotFoundError`.

---

## 4. Cách chạy

### 4.1 Notebook phần Model (Member A) — Step 5-9

**Mở notebook:** `notebooks/model_pipeline.ipynb` trên Jupyter hoặc Colab, chạy lần lượt từng cell từ trên xuống.

**Hoặc chạy bằng terminal:**
```bash
# Jupyter
jupyter notebook notebooks/model_pipeline.ipynb

# Colab: upload repo lên Colab, chạy lệnh ở cell đầu
!git clone <repo_url>
%cd SIMC_2026
!pip install -r requirements.txt
```

**Output sau khi chạy xong:**
- `notebooks/best_model.pkl` — pipeline (StandardScaler + XGBClassifier)
- `notebooks/feature_columns.pkl` — danh sách 55 cột feature (app.py dùng file này để predict đúng)
- `notebooks/model_comparison.csv` — bảng kết quả so sánh 5 model

### 4.2 Dashboard / App (Member C) — Step 4 + Integrating AI Services

**Trước tiên:** giải nén data (xem bước chuẩn bị ở mục 3).

**Chạy app:**
```bash
python app/app.py
```

Mở trình duyệt vào `http://127.0.0.1:8050`.

**Các ô nhập trên app:**
- Nhiệt độ (temperature)
- Tổng thiết bị đang bật (total_appliance, kW)
- Nhấn "Dự đoán và hỏi AI" → model dự đoán bất thường/bình thường + AI giải thích

**AI giải thích:**
- Nếu có biến môi trường `GEMINI_API_KEY` → gọi Gemini API thật
- Không có key → dùng fallback rule-based (vẫn chạy mượt, không sập khi demo)

Để đặt API key trên **Windows:**
```bash
set GEMINI_API_KEY=key_cua_ban
python app/app.py
```

### 4.3 EDA / Data Understanding (Member B) — Step 1-3

Chạy script tiền xử lý và EDA:
```bash
python src/HomeC_preprocess.py
```

Sẽ sinh ra:
- `data/HomeC_cleaned_final.csv` — dữ liệu đã làm sạch
- `visualization/chart*.png` — 4 biểu đồ

---

## 5. ⚙️ DEV_MODE — đọc kỹ phần này

Trong notebook `model_pipeline.ipynb` có biến `DEV_MODE` ở **SECTION 1**:

| Giá trị | Ý nghĩa | Khi nào dùng |
| --- | --- | --- |
| `DEV_MODE = True`  | Chỉ lấy mẫu 100.000 dòng → chạy **nhanh** (~10 phút) | Khi đang viết/sửa code, test thử cho lẹ |
| `DEV_MODE = False` | Dùng **toàn bộ ~500.000 dòng** → chạy **chậm** (~1-2 giờ) | Khi train bản cuối để lưu model nộp bài |

**Quy trình đúng:** Để `True` trong lúc dev cho nhanh → khi mọi thứ ổn, đổi sang `False` và chạy lại 1 lần cuối để lấy model + số liệu chính thức.

> ⚠️ **Số liệu từ DEV_MODE=True là bản nháp.** Nếu nộp bài với kết quả từ mẫu 100k dòng mà giảng viên kiểm tra lại sẽ thấy khác. Phải đảm bảo file `best_model.pkl` được train từ `DEV_MODE = False`.

> ⚠️ **Lưu ý tải máy:** Giữ nguyên 500k dòng nên **KHÔNG dùng SVM/One-Class SVM** (rất chậm, dễ treo máy). Các model trong pipeline (LogisticRegression, RandomForest, XGBoost, LightGBM, IsolationForest) đều chịu được khối lượng này. RandomForest chậy nhất (~20 phút tuning), XGBoost/LightGBM nhanh hơn.

> ⚠️ **Data leakage:** Không thêm cột `use [kW]` vào feature. Nhãn `anomaly` sinh ra từ chính cột này; nếu giữ nó làm feature sẽ bị leakage (F1 giả ≈ 1.0). Pipeline đã drop sẵn cột này.

---

## 5b. Kiến trúc ứng dụng

```
Data thô (HomeC.csv) 
    ↓
[Step 1-3: Data & EDA]  Member B → HomeC_cleaned_final.csv
    ↓
[Step 4: Feature Engineering]  Member C → feature_columns.pkl
    ↓
[Step 5-8: Train & Tune Model]  Member A → best_model.pkl + feature_columns.pkl
    ↓
[Ứng dụng Plotly Dash]
    ├─ UI: nhập thông số
    ├─ Model: dự đoán bất thường/bình thường
    └─ AI: giải thích kết quả (Gemini API hoặc fallback rule-based)
```

---

| Thành viên | Vai trò | Nội dung |
| --- | --- | --- |
| **A** | Model Lead | Tạo nhãn, Step 5–9 (split, train ≥5 model, tuning, pipeline) |
| **B** | Data & EDA | Step 1–3 (problem, RQ, metric, data understanding, EDA), báo cáo |
| **C** | Feature & Dashboard | Step 4 (feature engineering), app/dashboard |

Chi tiết theo ngày: xem file phân công công việc của nhóm.

---

## 9. Kết quả model (DEV_MODE = False, full ~500k dòng)

| Model | F1 | AUC | Tuning time |
| --- | --- | --- | --- |
| **XGBoost** | ~0.899 | ~0.998 | ~8 phút |
| **RandomForest** | ~0.890 | ~0.994 | ~20 phút |
| LightGBM | ~0.885 | ~0.996 | ~5 phút |
| LogisticRegression | ~0.37 | ~0.965 | <1 phút |
| IsolationForest (unsupervised) | ~0.08 | ~0.727 | <1 phút |

**Ghi chú:**
- **XGBoost & RandomForest** là hai model tốt nhất và tương đương.
- **LightGBM** chậy hơn XGBoost nhưng cũng đủ tốt và nhanh nhất trong top 3.
- **LogisticRegression** yếu vì dữ liệu có quan hệ phi tuyến và rất mất cân bằng.
- **IsolationForest** yếu hơn hẳn (unsupervised, không học từ proxy label) — là điểm thảo luận tốt khi so sánh supervised vs unsupervised trong Q&A.

**Anomaly ratio:** ~2.46% (khoảng 12,500 dòng trong ~500k).

---

## 10. Ghi chú trung thực (đọc trước khi báo cáo/Q&A)

### Proxy label
- F1 cao vì model học lại quy luật thống kê do chính nhóm đặt ra (proxy label),
  không phải phát hiện bất thường "thật" theo nghĩa nghiệp vụ.
- Khi giảng viên hỏi "nhãn lấy ở đâu ra" → trả lời thẳng: proxy label theo
  ngưỡng `mean + 3·std`, có trích baseline paper, và **nêu rõ giới hạn**.

### ⚠️ Lỗi timestamp dữ liệu
- Dataset HomeC được mô tả là "đo mỗi phút trong năm 2016" nhưng cột `time` Unix thực tế tăng **mỗi giây**.
- Hệ quả: dữ liệu chỉ trải ~6 ngày (1/1 - 7/1/2016) thay vì cả năm.
- **Feature `month`, `dayofweek`, `is_weekend` gần như vô dụng** (toàn bộ tập rơi vào 1 tuần).
- Feature `hour` vẫn có ý nghĩa (chu kỳ trong ngày vẫn còn).

**Cách xử lý:** Nhóm ghi chú giới hạn này trong phần Data Understanding của báo cáo. Giảng viên sẽ quyết định có loại bỏ các feature theo mùa hay giữ + ghi chú. Hiện tại các feature đó vẫn có trong `feature_columns.pkl` nhưng chúng chỉ học 1 giá trị duy nhất.

### Feature Engineering & SMOTE
- Step 4 (Member C) tạo feature: `hour`, `dayofweek`, `month`, `is_weekend`, `total_appliance`, one-hot encode `icon` & `summary`.
- Tất cả feature được **scale bằng StandardScaler** trong pipeline.
- SMOTE áp dụng **chỉ trên tập train** để tránh data leakage.

### Semantic (nếu nộp SIMC)
- Phần semantic hiện chỉ được nhắc nhẹ ở phần viết báo cáo.
- Đóng góp semantic còn mỏng so với yêu cầu của hội nghị Scopus.

---

## 11. Lưu ý pháp lý / học vụ

Mọi quyết định về điểm số, tính hợp lệ của dự án backup, hay lo ngại pháp lý của
dự án cũ nên xác nhận trực tiếp với **giảng viên hướng dẫn** và **bộ phận học vụ**.
