# Báo cáo sửa lỗi dự án `SIMC_2026_revised`

## 1. Kết quả thực hiện

Dự án đã được sửa trực tiếp từ file `SIMC_2026_revised.zip`, chạy lại preprocessing, tái tạo toàn bộ bảng KPI và 14 biểu đồ, huấn luyện lại model, cập nhật dashboard, notebook, README, báo cáo nội bộ và sơ đồ kiến trúc.

Bản sửa cuối gồm **34 file**, tổng dung lượng khoảng **27 MB**. Dataset tiếp tục được lưu dạng ZIP nên không làm gói nộp tăng lên hơn 160 MB.

## 2. Các lỗi đã sửa

### 2.1. Sửa sai đơn vị kW, kWh và MWh

Dữ liệu là công suất trung bình theo từng phút, đơn vị kW. Bản cũ cộng trực tiếp các giá trị kW rồi ghi thành kWh/MWh, khiến nhiều số liệu lớn hơn thực tế khoảng 60 lần.

Công thức đúng đã được áp dụng:

```text
Energy (kWh) = sum(Power kW) / 60
Energy (MWh) = sum(Power kW) / 60 / 1000
```

Các thành phần đã được sửa:

- `data/top5_appliances.csv`;
- `data/kpi_summary.json`;
- biểu đồ tổng năng lượng theo ngày;
- area chart Use vs Solar;
- stacked area theo tháng;
- calendar heatmap;
- Top-5 bar chart và pie chart;
- toàn bộ biểu đồ tương ứng trên dashboard.

### 2.2. Sửa lỗi cộng lặp Kitchen và Furnace

Bản cũ có thể đọc lại `HomeC_cleaned_final.csv`, trong đó đã tồn tại:

- `Kitchen [kW]`;
- `Furnace [kW]`;
- `total_appliance`;
- các feature thời gian.

Sau đó script lại đưa các cột này vào quá trình cộng nhóm, làm Furnace/Kitchen bị cộng thêm lần nữa.

Bản sửa loại toàn bộ feature engineered cũ trước khi tái tạo:

```text
hour, dayofweek, month, is_weekend, season, time_period,
Kitchen [kW], Furnace [kW], total_appliance
```

Các nhóm sau đó chỉ được tính từ 14 circuit gốc. Preprocessing đã được chạy hai lần liên tiếp và cho kết quả `top5_appliances.csv` cùng `kpi_summary.json` giống hệt nhau.

### 2.3. Không cần giải nén dataset thủ công

Các file sau giờ đọc trực tiếp:

```text
data/HomeC_cleaned_final.zip
```

- `src/HomeC_preprocess.py`;
- `src/train_model.py`;
- `app/app.py`;
- `notebooks/model_pipeline.ipynb`.

Preprocessing cũng ghi kết quả trở lại chính file ZIP theo cách an toàn bằng file tạm rồi thay thế.

### 2.4. Sửa phương pháp chia dữ liệu model

Bản cũ dùng `train_test_split(..., stratify=y)`, khiến các thời điểm trong cả năm bị trộn ngẫu nhiên giữa train và test. Với dữ liệu chuỗi thời gian và anomaly tập trung mạnh ở tháng 7–8, cách này cho kết quả quá lạc quan.

Bản sửa dùng chronological split:

| Tập | Tỷ lệ | Khoảng thời gian tái dựng |
|---|---:|---|
| Train | 70% | 2016-01-01 → 2016-09-01 |
| Validation | 15% | 2016-09-01 → 2016-10-24 |
| Test | 15% | 2016-10-24 → 2016-12-15 |

Model và decision threshold chỉ được chọn trên validation. Test set chỉ được dùng sau khi model cuối đã được chốt.

### 2.5. Loại SMOTE khỏi pipeline thời gian

SMOTE không còn được áp dụng. Các mẫu tổng hợp có thể phá vỡ cấu trúc thời gian và làm đánh giá khó diễn giải. Class imbalance được xử lý bằng class weight/scale weight và hiệu chỉnh decision threshold trên validation.

### 2.6. Giảm feature và đồng bộ với dashboard

Bản cũ train 60 feature nhưng dashboard chỉ nhập ba giá trị, sau đó điền 0 cho phần còn lại. Dự đoán vì vậy không phản ánh đầu vào thật.

Model mới dùng đúng tám feature deployable:

```text
gen_kw
total_appliance
temperature
humidity
hour
dayofweek
month
is_weekend
```

Dashboard hiện cho nhập đầy đủ:

- nhiệt độ;
- độ ẩm;
- solar generation;
- tổng công suất thiết bị;
- giờ;
- thứ trong tuần;
- tháng.

`is_weekend` được suy ra tự động từ `dayofweek`.

### 2.7. Sửa artifact model không khớp kết quả đánh giá

Bản cũ:

1. đánh giá model trên dữ liệu đã scale + SMOTE;
2. tuning làm Random Forest giảm F1 từ 0,9412 xuống 0,8477;
3. sau đó tạo pipeline mới và fit lại bằng cách khác;
4. vẫn lưu model đó làm `best_model.pkl`.

Bản sửa lưu đồng bộ:

- `best_model.pkl`;
- `feature_columns.pkl`;
- `feature_defaults.json`;
- `model_metadata.json`;
- `model_comparison.csv`.

`model_metadata.json` lưu rõ model, threshold, split, test metrics, confusion matrix và giới hạn nghiên cứu. Dashboard đọc trực tiếp metadata này.

### 2.8. Sửa dashboard

Các thay đổi chính trong `app/app.py`:

- đọc dữ liệu ZIP trực tiếp;
- quy đổi daily/monthly energy đúng đơn vị;
- dùng Top-5 mới;
- đọc model metadata và feature defaults;
- hiển thị tên model, số feature và test F1;
- dùng `predict_proba` cùng calibrated decision threshold;
- hiển thị anomaly score và threshold;
- không còn điền 0 cho hàng chục feature bị thiếu;
- đổi KPI `Average Energy` thành `Average Power`;
- sửa hướng dẫn chạy và thông báo lỗi dataset.

### 2.9. Bổ sung dependency

`requirements.txt` đã bổ sung các thư viện bị thiếu:

- `seaborn`;
- `python-dotenv`;
- `nbformat`.

### 2.10. Đồng bộ tài liệu

Đã cập nhật hoặc tạo lại:

- `README.md`;
- `report/Day2_MemberB_Report.md`;
- `report/architecture_pipeline_mermaid_source.md`;
- `report/architecture.png`;
- `report/architecture.svg`;
- `notebooks/model_pipeline.ipynb`.

Các tài liệu mới không còn ghi artifact bị xóa, không còn yêu cầu giải nén RAR/CSV và không còn dùng số liệu Furnace sai.

## 3. Số liệu dữ liệu sau khi sửa

| Chỉ số | Giá trị |
|---|---:|
| Số dòng | 503.910 |
| Số cột | 40 |
| Khoảng thời gian tái dựng | 2016-01-01 00:00 → 2016-12-15 22:29 |
| Tổng điện hộ gia đình | 7.214,00 kWh |
| Công suất trung bình | 0,8590 kW |
| Peak | 14,7146 kW |
| Tổng solar generation | 640,21 kWh |
| Proxy anomaly threshold | 4,0336 kW |
| Proxy anomalies | 12.418 |
| Proxy anomaly rate | 2,46% |

### Top-5 appliance đúng

| Appliance | Total energy | Share | Peak hour | Peak date |
|---|---:|---:|---:|---|
| Furnace | 1.981,96 kWh | 39,41% | 05:00 | 2016-02-14 |
| Home office | 682,69 kWh | 13,58% | 21:00 | 2016-08-09 |
| Fridge | 533,78 kWh | 10,62% | 20:00 | 2016-08-14 |
| Barn | 491,56 kWh | 9,78% | 16:00 | 2016-09-11 |
| Wine cellar | 353,88 kWh | 7,04% | 17:00 | 2016-08-14 |

## 4. Kết quả Research Questions sau khi kiểm tra lại

### RQ1 — Thời điểm anomaly xuất hiện nhiều

- 15:00 có anomaly rate cao nhất: khoảng **6,75%**.
- 16:00 và 17:00 lần lượt khoảng **5,83%** và **5,80%**.
- August có anomaly rate khoảng **11,91%**.
- July có anomaly rate khoảng **9,87%**.
- Monday có anomaly rate cao nhất theo thứ: khoảng **3,79%**.

### RQ2 — Appliance đóng góp chính

- Furnace chiếm **39,41%** tổng năng lượng của các appliance được đo.
- Furnace là appliance có giá trị lớn nhất trong **10.332/12.418** anomaly rows.

Điều này cho thấy Furnace liên quan mạnh đến proxy high-load anomaly, nhưng không đồng nghĩa thiết bị bị hỏng.

### RQ3 — Weather và energy

- Weather có tương quan tuyến tính yếu với household use ở mức daily aggregation.
- Solar generation tương quan dương với temperature, `r ≈ 0,356`.
- Nhiệt độ trung bình trong anomaly rows khoảng **68,25°F**, so với **50,30°F** ở normal rows.

Đây là association mô tả, chưa phải bằng chứng nhân quả.

## 5. Kết quả model mới

### Validation comparison

| Model | F1 | AUC | Precision | Recall | Decision threshold |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0,2987 | 0,9511 | 0,2241 | 0,4477 | 0,450795 |
| Random Forest | 0,2424 | 0,9287 | 0,1445 | 0,7503 | 0,139608 |
| XGBoost | 0,2222 | 0,9435 | 0,1427 | 0,5017 | 0,644966 |
| Logistic Regression | 0,2131 | 0,9230 | 0,9304 | 0,1204 | 0,999999 |
| Isolation Forest | 0,1381 | 0,7529 | 0,1453 | 0,1316 | 0,602938 |

### Selected final model

Selected model: **LightGBM**

Held-out chronological test result:

| Metric | Value |
|---|---:|
| F1 | 0,2912 |
| ROC-AUC | 0,8987 |
| Precision | 0,2648 |
| Recall | 0,3235 |

Confusion matrix:

```text
[[74748, 397],
 [  299, 143]]
```

Kết quả thấp hơn random split cũ nhưng đáng tin cậy hơn vì model phải dự đoán một giai đoạn tương lai chưa xuất hiện trong train.

## 6. Các file quan trọng đã sửa hoặc bổ sung

| File | Thay đổi |
|---|---|
| `src/HomeC_preprocess.py` | Viết lại preprocessing idempotent, đọc/ghi ZIP, sửa đơn vị, tạo lại KPI và 14 chart |
| `src/train_model.py` | Thêm pipeline chronological split, model comparison, threshold calibration và artifact export |
| `app/app.py` | Sửa dữ liệu/đơn vị, model input, prediction threshold và metadata |
| `notebooks/model_pipeline.ipynb` | Đồng bộ workflow và kết quả mới |
| `notebooks/best_model.pkl` | LightGBM model mới |
| `notebooks/feature_columns.pkl` | 8 feature đúng với dashboard |
| `notebooks/feature_defaults.json` | Default values khi cần |
| `notebooks/model_metadata.json` | Split, threshold, test metrics và limitations |
| `notebooks/model_comparison.csv` | Validation comparison mới |
| `data/HomeC_cleaned_final.zip` | Dataset 40 cột được tái tạo sạch, không double-count |
| `data/kpi_summary.json` | KPI đúng đơn vị |
| `data/top5_appliances.csv` | Top-5 đúng đơn vị và không cộng lặp |
| `visualization/*.png` | 14 hình được tạo lại |
| `README.md` | Hướng dẫn và kết quả mới |
| `report/Day2_MemberB_Report.md` | Phân tích RQ và giới hạn mới |
| `report/architecture.*` | Kiến trúc pipeline mới |
| `requirements.txt` | Bổ sung dependency thiếu |

## 7. Kiểm tra chất lượng đã thực hiện

- `HomeC_preprocess.py`: vượt qua `py_compile`.
- `train_model.py`: vượt qua `py_compile`.
- `app.py`: vượt qua `py_compile`.
- Notebook: hợp lệ theo chuẩn nbformat v4, gồm 13 cells.
- Dataset ZIP: đọc thành công, đúng 503.910 × 40.
- Dataset ZIP: vượt qua kiểm tra integrity.
- Kitchen group: khớp tổng ba circuit, sai số floating-point dưới `1e-15`.
- Furnace group: khớp tổng hai circuit, sai số dưới `1e-15`.
- `total_appliance`: khớp tổng 14 circuit, sai số dưới `1e-15`.
- Preprocessing chạy hai lần liên tiếp: Top-5 và KPI không thay đổi.
- Model artifact: load thành công.
- Feature count: đúng 8.
- Prediction thử với median defaults: thành công.
- Project không chứa CSV 162 MB đã giải nén và không chứa `__pycache__`/`.pyc`.

Dashboard chưa được mở bằng web server trong môi trường kiểm tra vì package Dash không được cài sẵn tại runtime hiện tại. Tuy nhiên file đã vượt qua kiểm tra cú pháp, model prediction logic đã được kiểm tra riêng, và `dash` đã có trong `requirements.txt`.

## 8. Cách chạy bản sửa

```bash
pip install -r requirements.txt
python src/HomeC_preprocess.py
python src/train_model.py
python app/app.py
```

Không cần giải nén `HomeC_cleaned_final.zip`.

## 9. Giới hạn còn lại

Các vấn đề sau là giới hạn nghiên cứu, không phải lỗi code có thể sửa hoàn toàn trong bản hiện tại:

1. anomaly label vẫn là proxy `mean + 3×std`, chưa phải ground-truth fraud/fault label;
2. timeline được tái dựng từ giả định một mẫu/phút;
3. `total_appliance` liên quan mạnh với household use và có thể chi phối model;
4. dữ liệu chỉ thuộc một household;
5. chronological test F1 cho thấy distribution shift đáng kể;
6. cần thêm rolling/walk-forward validation, verified event labels và external household validation nếu muốn phát triển thành paper/hội nghị.
