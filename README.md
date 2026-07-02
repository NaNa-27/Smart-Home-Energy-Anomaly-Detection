# SIMC_2026 — Anomaly Detection on Smart Home Energy Data

Đồ án môn **DAP391m** (dự án backup). Phát hiện bất thường trong tiêu thụ điện
của nhà thông minh, dựa trên dataset **HomeC (Smart Home Dataset with Weather
Information)**. Có thể chỉnh để nộp hội nghị **SIMC 2026** (mục tiêu phụ).

---

## 0. 🔧 NHẬT KÝ CHỈNH SỬA (đọc trước tiên)

Bản này đã sửa lỗi nghiêm trọng nhất của dự án + nâng cấp toàn bộ phần EDA & dashboard:

1. **[FIXED] Lỗi chỉ đọc được 01/01 → 07/01/2016.**
   Nguyên nhân: cột `time` (Unix) tăng **1 giây/dòng** chứ không phải 1 phút/dòng
   như mô tả của dataset, nên `pd.to_datetime(time, unit="s")` co cụm toàn bộ
   503,910 dòng vào ~5.8 ngày. `src/HomeC_preprocess.py` giờ bỏ qua cột `time`
   khi dựng lại mốc thời gian, và dùng `pd.date_range(start="2016-01-01",
   periods=len(df), freq="1min")` → trải đúng **2016-01-01 → 2016-12-15**
   (~350 ngày, khớp với mô tả "đo mỗi phút trong năm 2016").
2. **[FIXED] File input bị đặt sai tên.** Script cũ đọc `data/HomeC_cleaned.csv`
   — file này **không tồn tại** trong repo (chỉ có `HomeC_cleaned_final.csv`/`.rar`).
   Script mới tự dò: ưu tiên `data/HomeC.csv` (raw gốc, nếu nhóm có), nếu không có
   thì dùng `data/HomeC_cleaned_final.csv` đã có sẵn.
3. **[NEW]** Feature engineering đầy đủ: `hour`, `dayofweek`, `month`,
   `is_weekend`, `season`, `time_period`, `total_appliance`, và 2 cột gộp
   `Kitchen [kW]` / `Furnace [kW]`. Các feature này nay **có ý nghĩa thật**
   (trước đây `month`/`dayofweek`/`is_weekend` gần như vô dụng vì dữ liệu chỉ
   trải 1 tuần).
4. **[NEW]** Top-5 appliance analysis, weather↔energy correlation, time-series
   analysis (hourly/daily/day-of-week/monthly/calendar heatmap), anomaly
   visualization — xem mục 6.
5. **[NEW]** `data/kpi_summary.json`, `data/top5_appliances.csv`,
   `data/weather_energy_correlation.csv` — dashboard (`app/app.py`) đọc trực
   tiếp các file này cho KPI Cards & Alert Panel, không hardcode số liệu.
6. **[NEW]** Dashboard (`app/app.py`) bổ sung KPI Cards, Alert Panel, 10 biểu
   đồ Plotly tương tác, và panel Smart City Recommendation — xem mục 7.
7. **[BREAKING] `notebooks/best_model.pkl`, `feature_columns.pkl`,
   `model_comparison.csv` đã bị XOÁ khỏi bản này** vì chúng được train trên
   dữ liệu cũ bị lỗi (chỉ 6 ngày). Bảng kết quả model ở mục 9 (bản cũ) **không
   còn đúng** — phải chạy lại `notebooks/model_pipeline.ipynb` trên dữ liệu đã
   sửa để có số liệu thật. Đã kiểm tra: notebook chạy được, không lỗi, với
   `total_appliance` giờ thực sự là 1 feature hợp lệ (trước đây bị "câm" vì
   không tồn tại trong dữ liệu, dù `app.py` đã giả định có).
8. **[CHANGED]** Dataset nén bằng `.zip` thay vì `.rar` (không cần WinRAR/7-Zip
   để giải nén, mọi OS đều hỗ trợ `.zip` sẵn).

---

## 1. Tổng quan

- **Bài toán:** phát hiện các khoảng thời gian tiêu thụ điện bất thường.
- **Dataset:** HomeC — 503,910 dòng × 31 cột gốc (40 cột sau feature
  engineering), đo mỗi phút, **trải toàn bộ 01/2016 → giữa 12/2016** (đã sửa
  lỗi timestamp), gồm điện năng theo thiết bị + dữ liệu thời tiết.
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
│   ├── HomeC_cleaned_final.zip          # dữ liệu đã làm sạch + feature engineering (CHƯA GIẢI NÉN)
│   ├── kpi_summary.json                 # KPI tổng quan (dashboard đọc trực tiếp)
│   ├── top5_appliances.csv              # bảng Top-5 thiết bị (kWh, %, peak hour/date)
│   └── weather_energy_correlation.csv   # correlation thời tiết ↔ năng lượng
├── notebooks/
│   └── model_pipeline.ipynb       # phần A: tạo nhãn + train model (Step 5-9)
│                                   # (best_model.pkl / feature_columns.pkl / model_comparison.csv
│                                   #  cần CHẠY LẠI trên dữ liệu mới — xem mục 0.7)
├── src/
│   └── HomeC_preprocess.py        # phần B: tiền xử lý + EDA (Step 1-3) — ĐÃ SỬA TOÀN BỘ
├── app/
│   └── app.py                     # phần C: dashboard Plotly Dash + KPI/Alert/AI — ĐÃ NÂNG CẤP
├── report/
│   └── Day2_MemberB_Report.md     # phần B: problem, RQ, EDA, data understanding
├── visualization/
│   └── chart1.png … chart14.png   # 14 biểu đồ (xem danh sách ở mục 6)
├── requirements.txt                # danh sách thư viện (dùng pip install -r) — không đổi
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
Dataset được nén dưới dạng `.zip`. Trước khi chạy notebook hoặc app, bạn cần giải nén:

1. Chuột phải vào `data/HomeC_cleaned_final.zip` → "Extract Here" (Windows/macOS/Linux
   đều hỗ trợ `.zip` sẵn, không cần cài thêm phần mềm).
2. Sau giải nén, trong thư mục `data/` sẽ có file `HomeC_cleaned_final.csv` (~155MB).

Nếu chưa giải nén mà chạy notebook/app, sẽ báo lỗi `FileNotFoundError` (app.py sẽ
tự chuyển sang dữ liệu giả để demo không bị sập, nhưng số liệu sẽ không đúng).

---

## 4. Cách chạy

### 4.1 EDA / Data Understanding (Member B) — Step 1-3 — CHẠY TRƯỚC TIÊN

```bash
python src/HomeC_preprocess.py
```

Script sẽ:
- Tự dò input (`data/HomeC.csv` raw nếu có, nếu không dùng `HomeC_cleaned_final.csv`).
- Sửa lỗi timestamp → dữ liệu trải đúng cả năm 2016.
- Làm sạch (missing/duplicate/constant column), feature engineering, feature selection.
- Sinh Top-5 appliance, weather correlation, time-series, anomaly visualization.
- Lưu lại `data/HomeC_cleaned_final.csv` (40 cột, đã thêm feature mới),
  `data/kpi_summary.json`, `data/top5_appliances.csv`,
  `data/weather_energy_correlation.csv`, và 14 biểu đồ vào `visualization/`.

> Chạy script này TRƯỚC khi chạy notebook hoặc app, vì cả hai đều phụ thuộc vào
> `data/HomeC_cleaned_final.csv` đã được sửa lỗi + thêm feature.

### 4.2 Notebook phần Model (Member A) — Step 5-9

**Mở notebook:** `notebooks/model_pipeline.ipynb` trên Jupyter hoặc Colab, chạy lần lượt từng cell từ trên xuống.

```bash
jupyter notebook notebooks/model_pipeline.ipynb
```

**Output sau khi chạy xong:**
- `notebooks/best_model.pkl` — pipeline (StandardScaler + model tốt nhất)
- `notebooks/feature_columns.pkl` — danh sách cột feature (app.py dùng file này để predict đúng;
  giờ sẽ bao gồm cả `total_appliance`, `season`, `time_period`, `Kitchen [kW]`, `Furnace [kW]`)
- `notebooks/model_comparison.csv` — bảng kết quả so sánh model

> ⚠️ Đã kiểm tra logic notebook trên một mẫu nhỏ của dữ liệu mới: chạy được hết,
> không lỗi cột/kiểu dữ liệu. Số liệu F1/AUC ở bản DEV_MODE=False **phải chạy
> lại từ đầu** vì dữ liệu nền (full năm thay vì 6 ngày) đã khác hẳn bản cũ.

### 4.3 Dashboard / App (Member C) — Step 4 + Integrating AI Services

```bash
python app/app.py
```

Mở trình duyệt vào `http://127.0.0.1:8050`.

**Dashboard hiện có:**
- **KPI Cards:** Total Energy, Average Energy, Peak Consumption, Number of Anomalies
  (đọc từ `data/kpi_summary.json`).
- **Alert Panel:** danh sách các thời điểm bất thường nổi bật nhất, kèm thiết
  bị đóng góp nhiều nhất tại thời điểm đó.
- **10 biểu đồ tương tác:** Line (tiêu thụ theo ngày), Area (use vs gen), Stacked
  Area (top thiết bị theo tháng), Correlation Heatmap, Weather Scatter +
  Regression (chọn cặp biến qua dropdown), Top-5 Bar, Pie, Hourly Bar, Day-of-week
  Bar, Calendar Heatmap.
- **Smart City Recommendation:** gợi ý tiết kiệm điện sinh ra từ dữ liệu thật
  (thiết bị chiếm tỷ trọng cao nhất, giờ cao điểm, tương quan thời tiết/solar...).
- **Dự đoán + AI giải thích** (giữ nguyên từ bản trước, mở rộng thêm input `hour`).

**AI giải thích:**
- Nếu có biến môi trường `GEMINI_API_KEY` → gọi Gemini API thật.
- Không có key → dùng fallback rule-based (vẫn chạy mượt, không sập khi demo).

Để đặt API key trên **Windows:**
```bash
set GEMINI_API_KEY=key_cua_ban
python app/app.py
```

---

## 5. ⚙️ DEV_MODE — đọc kỹ phần này

Trong notebook `model_pipeline.ipynb` có biến `DEV_MODE` ở **SECTION 1**:

| Giá trị | Ý nghĩa | Khi nào dùng |
| --- | --- | --- |
| `DEV_MODE = True`  | Chỉ lấy mẫu 100.000 dòng → chạy **nhanh** (~10 phút) | Khi đang viết/sửa code, test thử cho lẹ |
| `DEV_MODE = False` | Dùng **toàn bộ ~500.000 dòng** → chạy **chậm** (~1-2 giờ) | Khi train bản cuối để lưu model nộp bài |

**Quy trình đúng:** Để `True` trong lúc dev cho nhanh → khi mọi thứ ổn, đổi sang `False` và chạy lại 1 lần cuối để lấy model + số liệu chính thức.

> ⚠️ **Số liệu từ DEV_MODE=True là bản nháp.** Nếu nộp bài với kết quả từ mẫu 100k dòng mà giảng viên kiểm tra lại sẽ thấy khác. Phải đảm bảo file `best_model.pkl` được train từ `DEV_MODE = False`.

> ⚠️ **Lưu ý tải máy:** Giữ nguyên 500k dòng nên **KHÔNG dùng SVM/One-Class SVM** (rất chậm, dễ treo máy). Các model trong pipeline (LogisticRegression, RandomForest, XGBoost, LightGBM, IsolationForest) đều chịu được khối lượng này.

> ⚠️ **Data leakage:** Không thêm cột `use [kW]` vào feature. Nhãn `anomaly` sinh ra từ chính cột này; nếu giữ nó làm feature sẽ bị leakage (F1 giả ≈ 1.0). Pipeline đã drop sẵn cột này.

---

## 6. Danh sách 14 biểu đồ (visualization/)

| File | Nội dung |
| --- | --- |
| chart1_timeseries.png | Tiêu thụ điện theo ngày, cả năm 2016 (Line) |
| chart2_histogram.png | Phân phối tiêu thụ điện theo phút (Histogram) |
| chart3_boxplot.png | Boxplot + ngưỡng bất thường |
| chart4_heatmap.png | Correlation heatmap: energy + top appliances + weather |
| chart5_weather_scatter.png | Scatter + regression: thời tiết vs energy/solar (2×2) |
| chart6_top5_appliances_bar.png | Top-5 thiết bị tiêu thụ điện nhiều nhất |
| chart7_appliance_pie.png | Tỷ trọng tiêu thụ: Top-5 vs Others |
| chart8_hourly_consumption.png | Tiêu thụ TB theo giờ (màu Low/Medium/High) |
| chart9_daily_area.png | Area chart: Use vs Solar Generation theo ngày |
| chart10_stacked_area_appliances.png | Stacked area: top thiết bị theo tháng |
| chart11_calendar_heatmap.png | Calendar heatmap: tổng tiêu thụ theo ngày trong năm |
| chart12_anomaly_timeseries.png | Time series với điểm bất thường được đánh dấu |
| chart13_dow_consumption.png | Tiêu thụ TB theo ngày trong tuần |
| chart14_monthly_consumption.png | Tiêu thụ TB theo tháng (chỉ có ý nghĩa sau khi sửa lỗi timestamp) |

---

## 7. Kiến trúc ứng dụng

```
Data thô (HomeC.csv / HomeC_cleaned_final.csv)
    ↓
[Step 1-3: Data & EDA]  Member B → HomeC_cleaned_final.csv (đã sửa timestamp + feature mới)
                                  → kpi_summary.json, top5_appliances.csv,
                                    weather_energy_correlation.csv
                                  → 14 biểu đồ trong visualization/
    ↓
[Step 5-8: Train & Tune Model]  Member A → best_model.pkl + feature_columns.pkl
    ↓
[Ứng dụng Plotly Dash]  Member C
    ├─ KPI Cards + Alert Panel (đọc trực tiếp kpi_summary.json)
    ├─ 10 biểu đồ tương tác (đồng bộ màu với visualization/*.png)
    ├─ Smart City Recommendation (sinh từ top5_appliances.csv + correlation)
    ├─ UI: nhập thông số → Model dự đoán bất thường/bình thường
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

## 8. Top-5 thiết bị tiêu thụ điện (tính trên dữ liệu đã sửa, cả năm 2016)

| Thiết bị | Tổng (kWh) | Tỷ trọng | Giờ hoạt động mạnh nhất |
| --- | --- | --- | --- |
| Furnace [kW] (gộp Furnace 1+2) | 118,917 | 39.41% | 5h |
| Home office [kW] | 40,961 | 13.58% | 21h |
| Fridge [kW] | 32,027 | 10.62% | 20h |
| Barn [kW] | 29,494 | 9.78% | 16h |
| Wine cellar [kW] | 21,233 | 7.04% | 17h |

Xem bảng đầy đủ tại `data/top5_appliances.csv` (gồm cả Peak_Date cho mỗi thiết bị).

**Weather ↔ Energy (correlation theo ngày):** thời tiết có tương quan **mạnh
với Solar Generation** (temperature: r≈0.36, dewPoint: r≈0.29) nhưng **yếu với
Use trực tiếp** (hành vi dùng thiết bị chi phối nhiều hơn thời tiết ở độ phân
giải 1 phút). Chi tiết: `data/weather_energy_correlation.csv`.

---

## 9. ⚠️ Kết quả model — CẦN CHẠY LẠI

Bảng kết quả model ở các bản trước (XGBoost F1≈0.899...) được train trên dữ
liệu **bị lỗi timestamp (chỉ 6 ngày)** và đã bị xoá khỏi bản này. Sau khi sửa
lỗi, không gian feature và phân phối dữ liệu đã thay đổi (đặc biệt `month`,
`dayofweek`, `is_weekend` giờ mới thực sự có biến thiên, và có thêm
`total_appliance`, `season`, `time_period`). **Hãy chạy lại
`notebooks/model_pipeline.ipynb` với `DEV_MODE = False`** để có bảng F1/AUC
chính thức trước khi báo cáo.

**Anomaly ratio (đã kiểm tra lại trên dữ liệu mới):** ~2.46% (12,418 / 503,910 dòng).

---

## 10. Ghi chú trung thực (đọc trước khi báo cáo/Q&A)

### Proxy label
- F1 cao vì model học lại quy luật thống kê do chính nhóm đặt ra (proxy label),
  không phải phát hiện bất thường "thật" theo nghĩa nghiệp vụ.
- Khi giảng viên hỏi "nhãn lấy ở đâu ra" → trả lời thẳng: proxy label theo
  ngưỡng `mean + 3·std`, có trích baseline paper, và **nêu rõ giới hạn**.

### ✅ Lỗi timestamp dữ liệu — ĐÃ SỬA
- Xem chi tiết nguyên nhân & cách sửa ở mục 0.1. Toàn bộ feature theo thời
  gian (`month`, `dayofweek`, `is_weekend`, `season`) nay có biến thiên thật.

### Feature Engineering & SMOTE
- `src/HomeC_preprocess.py` (Member B) tạo feature: `hour`, `dayofweek`,
  `month`, `is_weekend`, `season`, `time_period`, `total_appliance`, gộp
  `Kitchen [kW]` / `Furnace [kW]`, one-hot encode `icon` & `summary` (ở notebook).
- Tất cả feature được **scale bằng StandardScaler** trong pipeline.
- SMOTE áp dụng **chỉ trên tập train** để tránh data leakage.

### Semantic (nếu nộp SIMC)
- Phần semantic hiện chỉ được nhắc nhẹ ở phần viết báo cáo.
- Đóng góp semantic còn mỏng so với yêu cầu của hội nghị Scopus.

---

## 11. Lưu ý pháp lý / học vụ

Mọi quyết định về điểm số, tính hợp lệ của dự án backup, hay lo ngại pháp lý của
dự án cũ nên xác nhận trực tiếp với **giảng viên hướng dẫn** và **bộ phận học vụ**.
