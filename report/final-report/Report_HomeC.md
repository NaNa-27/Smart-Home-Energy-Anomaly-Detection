# BÁO CÁO DỰ ÁN

# Smart Home Energy Anomaly Detection on HomeC Dataset

## 1. Tóm tắt

Dự án này xây dựng một pipeline phát hiện bất thường trong tiêu thụ điện của nhà thông minh dựa trên bộ dữ liệu HomeC. Mục tiêu là nhận diện các thời điểm hộ gia đình có mức tiêu thụ điện cao bất thường từ dữ liệu công suất theo từng phút, dữ liệu phát điện mặt trời, thông tin thiết bị điện, đặc trưng thời gian và các biến thời tiết.

Trong quá trình rà soát và chỉnh sửa, dự án phát hiện một vấn đề quan trọng về phương pháp: nhãn anomaly ban đầu được tạo từ `use [kW]`, trong khi một đặc trưng đầu vào là `total_appliance` lại là thành phần cộng trực tiếp của chính `use [kW]`. Điều này tạo ra target leakage, tức là mô hình được cung cấp một phần thông tin gần như trực tiếp của đại lượng cần dự đoán.

Bản sửa đổi của dự án tập trung vào tính đúng đắn của thực nghiệm thay vì tối đa hóa điểm số một cách sai lệch. Dự án đã bổ sung baseline, tính ngưỡng global chỉ trên tập train, loại bỏ `total_appliance` khỏi mô hình deploy, sử dụng chia dữ liệu theo thời gian 70/15/15, hiệu chuẩn ngưỡng quyết định trên validation, và thêm nhãn chính dựa trên ngưỡng cục bộ 30 ngày trước đó.

Kết quả quan trọng nhất là baseline một cột `total_appliance` đạt test F1 = 0.4866, cao hơn toàn bộ các cấu hình machine learning sau khi chạy lại. Khi loại bỏ cột gây leakage, hiệu năng mô hình giảm mạnh. Điều này cho thấy đóng góp chính của dự án không nằm ở việc tạo ra một classifier mạnh, mà nằm ở việc chẩn đoán và sửa các vấn đề về leakage, seasonality và cách định nghĩa nhãn anomaly.

## 2. Giới thiệu

Trong hệ thống nhà thông minh, việc theo dõi và phát hiện bất thường trong tiêu thụ điện có thể giúp người dùng nhận biết các hành vi sử dụng năng lượng không hợp lý, thiết bị hoạt động ngoài dự kiến, hoặc nguy cơ sự cố điện. Tuy nhiên, bài toán phát hiện bất thường trong dữ liệu năng lượng không đơn giản. Một mức tiêu thụ cao chưa chắc là lỗi; nó có thể đến từ mùa hè, thói quen sinh hoạt, lịch làm việc của thiết bị, hoặc điều kiện thời tiết.

Dự án này sử dụng dữ liệu HomeC để xây dựng hệ thống phân tích và dự đoán anomaly trong tiêu thụ điện. Ban đầu, anomaly được định nghĩa bằng quy tắc 3-sigma trên `use [kW]`. Sau khi phân tích lại, dự án nhận thấy cách định nghĩa nhãn và bộ đặc trưng ban đầu có thể làm mô hình đạt kết quả cao nhưng không thật sự học được quy luật độc lập.

Các mục tiêu chính của dự án gồm:

1. Tiền xử lý và làm sạch dữ liệu HomeC.
2. Xây dựng các visualization để hiểu hành vi tiêu thụ điện.
3. Tạo nhãn anomaly theo cách ít sai lệch hơn.
4. Huấn luyện và so sánh các mô hình machine learning.
5. Kiểm tra baseline để đánh giá đóng góp thật của mô hình.
6. Xây dựng dashboard trực quan hóa KPI, cảnh báo và dự đoán anomaly.

## 3. Mô tả dữ liệu

Bộ dữ liệu HomeC được xử lý dưới dạng chuỗi thời gian với tần suất một phút. Do cột thời gian gốc trong dữ liệu tăng theo đơn vị 1 nhưng không thể dùng trực tiếp như Unix timestamp đáng tin cậy, dự án tái dựng timeline với giả định mỗi dòng là một bản ghi cách nhau một phút.

Thông tin tổng quan của dữ liệu sau xử lý:

| Chỉ số | Giá trị |
|---|---:|
| Khoảng thời gian | 2016-01-01 đến 2016-12-15 |
| Tần suất lấy mẫu | 1 phút |
| Tổng điện năng tiêu thụ | 7,214.00 kWh |
| Công suất trung bình | 0.859 kW |
| Công suất đỉnh | 14.7146 kW |
| Thời điểm công suất đỉnh | 2016-07-30 16:04:00 |
| Tổng điện mặt trời tạo ra | 640.21 kWh |
| Số anomaly theo nhãn local 30 ngày | 11,140 dòng |
| Tỷ lệ anomaly theo nhãn local 30 ngày | 2.21% |

Dữ liệu bao gồm các nhóm thông tin chính:

1. Công suất tiêu thụ toàn nhà: `use [kW]`.
2. Công suất điện mặt trời: `gen [kW]`.
3. Công suất của các thiết bị hoặc nhóm thiết bị.
4. Dữ liệu thời tiết như temperature, humidity, pressure, windSpeed, cloudCover, dewPoint.
5. Đặc trưng thời gian như hour, dayofweek, month, is_weekend.

## 4. Tiền xử lý dữ liệu

Pipeline tiền xử lý thực hiện các bước chính sau:

1. Đọc dữ liệu từ `HomeC.csv`, `HomeC_cleaned_final.csv` hoặc `HomeC_cleaned_final.zip`.
2. Chuẩn hóa tên cột và loại bỏ các cột engineered cũ để đảm bảo chạy lại nhiều lần vẫn nhất quán.
3. Tái dựng timeline theo tần suất một phút.
4. Xử lý missing values bằng median cho biến số và mode cho biến dạng text.
5. Loại bỏ dòng trùng, cột trùng và cột hằng.
6. Tạo đặc trưng thời gian gồm `hour`, `dayofweek`, `month`, `is_weekend`, `season`, `time_period`.
7. Gộp các nhóm thiết bị như kitchen và furnace.
8. Tính `total_appliance` để phân tích và chạy baseline, nhưng không dùng cột này trong mô hình deploy.

Điểm quan trọng nhất trong bản sửa là kiểm soát leakage. `total_appliance` không còn được xem là một đặc trưng hợp lệ cho mô hình cuối, vì nó là thành phần cộng của `use [kW]`, trong khi nhãn anomaly lại được tạo từ `use [kW]`.

## 5. Phân tích khám phá dữ liệu và visualization

Dự án tạo các biểu đồ tĩnh trong thư mục `visualization/` và các biểu đồ tương tác trong dashboard. Các visualization giúp giải thích phân bố dữ liệu, mức tiêu thụ theo thời gian, đóng góp của thiết bị và quan hệ với thời tiết.

Bảng dưới đây kiểm kê đầy đủ 14 biểu đồ tĩnh đã được sinh ra trong project:

| File visualization | Nội dung chính | Vai trò trong báo cáo |
|---|---|---|
| `chart1_timeseries.png` | Chuỗi thời gian tiêu thụ điện theo ngày | Nhận diện xu hướng và mùa vụ |
| `chart2_histogram.png` | Phân phối công suất tiêu thụ | Kiểm tra đuôi phải và giá trị cao bất thường |
| `chart3_boxplot.png` | Boxplot công suất tiêu thụ | Quan sát outlier và độ lệch phân phối |
| `chart4_heatmap.png` | Heatmap tương quan các biến | Đánh giá quan hệ giữa năng lượng, thiết bị và thời tiết |
| `chart5_weather_scatter.png` | Scatter giữa thời tiết và năng lượng | Minh họa tương quan yếu với `use [kW]` và rõ hơn với `gen [kW]` |
| `chart6_top5_appliances_bar.png` | Bar chart top thiết bị tiêu thụ điện | Xác định nhóm thiết bị đóng góp lớn nhất |
| `chart7_appliance_pie.png` | Tỷ trọng tiêu thụ của thiết bị | Thể hiện cơ cấu tiêu thụ theo thiết bị |
| `chart8_hourly_consumption.png` | Tiêu thụ trung bình theo giờ | Phân tích pattern sinh hoạt theo giờ |
| `chart9_daily_area.png` | Area chart tiêu thụ theo ngày | Theo dõi biến động tổng tải qua thời gian |
| `chart10_stacked_area_appliances.png` | Stacked area theo nhóm thiết bị | Quan sát đóng góp thiết bị theo thời gian |
| `chart11_calendar_heatmap.png` | Calendar heatmap tiêu thụ | Phát hiện ngày cao điểm và cụm bất thường |
| `chart12_anomaly_timeseries.png` | Timeline anomaly | Kiểm tra vị trí anomaly theo thời gian |
| `chart13_dow_consumption.png` | Tiêu thụ theo thứ trong tuần | So sánh weekday và weekend |
| `chart14_monthly_consumption.png` | Tiêu thụ theo tháng | Kiểm tra ảnh hưởng mùa vụ |

### 5.1 Tiêu thụ điện theo thời gian

Biểu đồ `chart1_timeseries.png` thể hiện tổng điện năng tiêu thụ theo ngày trong năm 2016. Đường thời gian cho thấy mức tiêu thụ không ổn định và có yếu tố mùa vụ. Các giai đoạn mùa hè có xu hướng xuất hiện nhiều mức tiêu thụ cao hơn, điều này ảnh hưởng trực tiếp đến cách định nghĩa anomaly nếu dùng một ngưỡng global cho cả năm.

### 5.2 Phân phối công suất tiêu thụ

Các biểu đồ `chart2_histogram.png` và `chart3_boxplot.png` cho thấy phần lớn giá trị công suất tập trung ở mức thấp, nhưng vẫn có một đuôi phải gồm các điểm tiêu thụ rất cao. Quy tắc 3-sigma có thể bắt được các điểm cực trị, nhưng nếu dùng global threshold thì có nguy cơ xem các đỉnh mùa vụ là anomaly.

### 5.3 Thiết bị tiêu thụ nhiều nhất

Năm thiết bị hoặc nhóm thiết bị tiêu thụ nhiều điện nhất:

| Thiết bị | Tổng kWh | Tỷ trọng | Giờ cao điểm | Ngày cao điểm |
|---|---:|---:|---:|---|
| Furnace | 1,981.96 | 39.41% | 05:00 | 2016-02-14 |
| Home office | 682.69 | 13.58% | 21:00 | 2016-08-09 |
| Fridge | 533.78 | 10.62% | 20:00 | 2016-08-14 |
| Barn | 491.56 | 9.78% | 16:00 | 2016-09-11 |
| Wine cellar | 353.88 | 7.04% | 17:00 | 2016-08-14 |

Biểu đồ `chart6_top5_appliances_bar.png` và `chart7_appliance_pie.png` cho thấy Furnace chiếm tỷ trọng lớn nhất, khoảng 39.41% tổng điện năng của các thiết bị. Vì vậy, nếu triển khai hệ thống gợi ý tiết kiệm điện, các thiết bị tiêu thụ cao như Furnace nên được ưu tiên phân tích trước.

### 5.4 Quan hệ giữa thời tiết và năng lượng

Bảng tương quan cho thấy các biến thời tiết có tương quan yếu với `use [kW]`, nhưng có tương quan rõ hơn với `gen [kW]`, đặc biệt là temperature và apparentTemperature.

| Biến thời tiết | Tương quan với `use [kW]` | Tương quan với `gen [kW]` |
|---|---:|---:|
| Temperature | 0.009 | 0.356 |
| Humidity | 0.077 | -0.079 |
| Cloud cover | -0.078 | -0.059 |
| Wind speed | -0.008 | -0.145 |
| Dew point | 0.038 | 0.287 |

Kết quả này cho thấy thời tiết có liên hệ với lượng điện mặt trời tạo ra hơn là trực tiếp giải thích mức tiêu thụ điện toàn nhà. Đây cũng là lý do các mô hình chỉ dùng biến thời tiết và thời gian không đạt F1 cao sau khi loại bỏ leakage.

### 5.5 Mẫu hình theo thời gian

Các biểu đồ `chart8_hourly_consumption.png`, `chart13_dow_consumption.png` và `chart14_monthly_consumption.png` thể hiện mức tiêu thụ trung bình theo giờ, thứ trong tuần và tháng. Các biểu đồ này cho thấy dữ liệu có pattern theo thời gian, nhưng pattern này chưa đủ mạnh để dự đoán anomaly chính xác nếu không có thêm đặc trưng lịch sử tiêu thụ.

## 6. Định nghĩa nhãn anomaly

### 6.1 Nhãn ban đầu

Cách định nghĩa ban đầu:

`anomaly = use [kW] > mean(use [kW]) + 3 * std(use [kW])`

Cách này có hai vấn đề:

1. Nếu mean và std được tính trên toàn bộ dữ liệu, thông tin từ validation và test đã đi vào quá trình tạo nhãn.
2. Với dữ liệu có mùa vụ, ngưỡng global có thể định nghĩa mùa hè là bất thường thay vì định nghĩa anomaly thật sự.

### 6.2 Nhãn global tính trên train

Bản sửa đầu tiên là chỉ tính ngưỡng global trên tập train:

`anomaly = use [kW] > mean(train use) + 3 * std(train use)`

Ngưỡng global train-only là 4.3952 kW. Cách này tránh leakage từ test vào ngưỡng, nhưng vẫn chưa giải quyết hoàn toàn vấn đề mùa vụ.

### 6.3 Nhãn local 30 ngày

Nhãn chính của bản sửa sử dụng ngưỡng cục bộ dựa trên 30 ngày trước đó:

`anomaly_t = use_t > rolling_mean(previous 30 days) + 3 * rolling_std(previous 30 days)`

Dòng hiện tại được loại khỏi cửa sổ tính ngưỡng. Vì vậy, nhãn này hỏi rằng mức tiêu thụ hiện tại có cao bất thường so với chính giai đoạn gần đây hay không, thay vì so với toàn bộ năm.

Tỷ lệ anomaly theo từng split:

| Cách định nghĩa nhãn | Train | Validation | Test |
|---|---:|---:|---:|
| Global 3-sigma tính trên train | 2.527% | 0.983% | 0.368% |
| Local 30-day 3-sigma | 2.505% | 0.693% | 2.355% |

Nhãn global tạo ra sự lệch phân phối lớn giữa train và test. Trong khi đó, nhãn local 30 ngày giúp tỷ lệ anomaly ở test trở nên hợp lý hơn và ít bị phụ thuộc vào mùa hè.

Tỷ lệ anomaly theo tháng:

| Tháng | Global label | Local 30-day label |
|---:|---:|---:|
| 1 | 0.361% | 1.396% |
| 2 | 0.584% | 1.830% |
| 3 | 0.291% | 1.228% |
| 4 | 0.292% | 1.748% |
| 5 | 0.361% | 1.427% |
| 6 | 0.218% | 1.803% |
| 7 | 8.107% | 7.807% |
| 8 | 9.810% | 2.789% |
| 9 | 1.512% | 0.204% |
| 10 | 0.302% | 1.810% |
| 11 | 0.421% | 2.528% |
| 12 | 0.237% | 1.469% |

Tháng 8 giảm từ 9.810% xuống 2.789% khi dùng nhãn local, cho thấy nhãn local đã giảm đáng kể ảnh hưởng mùa vụ. Tháng 7 vẫn còn cao ở mức 7.807%, điều này cho thấy tháng 7 có một chế độ tiêu thụ kéo dài mà cửa sổ 30 ngày chưa hấp thụ hết.

## 7. Đặc trưng và kiểm soát leakage

Bộ đặc trưng ban đầu gồm:

`gen [kW]`, `total_appliance`, `temperature`, `humidity`, `hour`, `dayofweek`, `month`, `is_weekend`

Bộ đặc trưng của mô hình deploy sau khi sửa:

`gen_kw`, `temperature`, `humidity`, `hour`, `dayofweek`, `month`, `is_weekend`

Lý do loại bỏ `total_appliance` là vì cột này có quan hệ trực tiếp với `use [kW]`. Nếu giữ cột này, mô hình không thật sự học bất thường từ thời tiết hoặc thời gian, mà chủ yếu học lại một phần của công thức tạo nhãn. Do đó, `total_appliance` chỉ được dùng để chạy baseline và chứng minh leakage.

## 8. Thiết lập thực nghiệm

Dữ liệu được chia theo thời gian:

| Tập dữ liệu | Tỷ lệ |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Chia theo thời gian phản ánh tình huống triển khai thực tế hơn so với chia ngẫu nhiên, vì hệ thống phải dự đoán dữ liệu tương lai từ dữ liệu quá khứ. Ngưỡng quyết định của mô hình được hiệu chuẩn trên validation và sau đó giữ cố định khi đánh giá trên test.

Dự án đánh giá các nhóm thực nghiệm:

1. Baseline majority normal.
2. Baseline một cột `total_appliance`.
3. Global label với bộ đặc trưng có leakage.
4. Global label sau khi loại leakage.
5. Local 30-day label sau khi loại leakage.

Các mô hình được thử gồm Logistic Regression, Random Forest, XGBoost, LightGBM và Isolation Forest.

## 9. Kết quả huấn luyện

### 9.1 Baseline

| Thực nghiệm | Test F1 | Test AUC | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|
| Majority normal | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| Baseline `total_appliance` | 0.4866 | 0.9235 | 0.9479 | 0.3273 | 4.0243 |

Baseline `total_appliance` đạt F1 cao nhất trong toàn bộ thực nghiệm. Điều này xác nhận rằng hiệu năng ban đầu của pipeline phụ thuộc mạnh vào một cột có quan hệ trực tiếp với nhãn.

### 9.2 Global label

| Thực nghiệm | Mô hình | Validation F1 | Test F1 | Test AUC | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Global label + leaky features | Logistic Regression | 0.2259 | 0.3931 | 0.9020 | 1.0000 | 0.2446 |
| Global label + leaky features | LightGBM | 0.2539 | 0.0000 | 0.8747 | 0.0000 | 0.0000 |
| Global label + leakage removed | Logistic Regression | 0.0235 | 0.0000 | 0.5992 | 0.0000 | 0.0000 |
| Global label + leakage removed | LightGBM | 0.0545 | 0.0000 | 0.6981 | 0.0000 | 0.0000 |

Khi còn leaky feature, Logistic Regression đạt test F1 = 0.3931. Sau khi loại `total_appliance`, F1 giảm về 0. Điều này cho thấy các biến còn lại không đủ mạnh để tái tạo hiệu năng ban đầu.

### 9.3 Local 30-day label và leakage-free features

| Mô hình | Validation F1 | Test F1 | Test AUC | Precision | Recall | Decision threshold |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0284 | 0.0000 | 0.6974 | 0.0000 | 0.0000 | 0.8208 |
| Random Forest | 0.0474 | 0.0839 | 0.6984 | 0.0515 | 0.2275 | 0.1955 |
| XGBoost | 0.1356 | 0.0000 | 0.7771 | 0.0000 | 0.0000 | 0.8318 |
| LightGBM | 0.0653 | 0.0795 | 0.7525 | 0.0600 | 0.1174 | 0.6925 |
| Isolation Forest | 0.0357 | 0.0316 | 0.4299 | 0.0202 | 0.0719 | 0.0075 |

XGBoost được chọn làm mô hình deploy vì có validation F1 cao nhất trong nhóm thực nghiệm chính. Tuy nhiên, trên test set, mô hình có F1 = 0.0000 dù AUC = 0.7771. Ma trận nhầm lẫn của XGBoost:

| | Dự đoán normal | Dự đoán anomaly |
|---|---:|---:|
| Thực tế normal | 73,807 | 0 |
| Thực tế anomaly | 1,780 | 0 |

Kết quả này cho thấy mô hình vẫn có khả năng xếp hạng ở mức nhất định, thể hiện qua AUC, nhưng ngưỡng quyết định được chọn trên validation không chuyển tốt sang test. Random Forest, LightGBM và Isolation Forest có F1 khác 0 trên test, nhưng mức F1 vẫn thấp. Vì vậy, không nên kết luận rằng mô hình đã đủ tốt để deploy anomaly detection thật sự.

### 9.4 Artifact huấn luyện và triển khai

Sau khi chạy `src/train_model.py`, project sinh ra các artifact chính trong thư mục `notebooks/`:

| File | Nội dung | Vai trò |
|---|---|---|
| `experiment_results.csv` | Toàn bộ baseline, leakage experiments và primary experiments | Bảng kết quả đầy đủ cho phần đánh giá |
| `model_comparison.csv` | Bảng so sánh 5 model chính | Nguồn số liệu cho dashboard và báo cáo ngắn |
| `model_metadata.json` | Model được chọn, threshold, test metrics, confusion matrix, split rates và limitations | Metadata để dashboard hiển thị trạng thái model |
| `best_model.pkl` | Model deploy đã được lưu bằng `joblib` | Artifact dự đoán trong dashboard |
| `feature_columns.pkl` | Danh sách 7 feature không leakage | Đảm bảo input prediction đúng thứ tự feature |
| `feature_defaults.json` | Giá trị median mặc định cho form dự đoán | Điền giá trị ban đầu trong dashboard |
| `model_pipeline.ipynb` | Notebook mô tả cách tái tạo kết quả | Tài liệu reproducibility cho người đọc |

Model deploy hiện tại là **XGBoost**, được chọn theo validation F1 cao nhất trong nhóm thực nghiệm chính. Bộ feature deploy gồm `gen_kw`, `temperature`, `humidity`, `hour`, `dayofweek`, `month` và `is_weekend`.

## 10. Dashboard và ứng dụng

Ứng dụng dashboard đọc dữ liệu từ `data/HomeC_cleaned_final.zip` và model artifacts từ thư mục `notebooks/`. Dashboard bao gồm:

1. KPI cards: tổng điện năng, công suất trung bình, công suất đỉnh và số anomaly.
2. Alert panel: hiển thị các cảnh báo anomaly dựa trên nhãn local 30 ngày.
3. Biểu đồ tương tác: daily consumption, solar generation, top appliances, hourly consumption và day-of-week consumption.
4. Form dự đoán anomaly sử dụng bộ đặc trưng không có leakage.
5. Phần giải thích bằng AI nếu người dùng tự cấu hình `GEMINI_API_KEY` trong môi trường local.

Project không đóng gói file `.env` hoặc API key. Ứng dụng cũng không in hậu tố key ra log, giúp giảm rủi ro lộ thông tin khi demo hoặc chia sẻ mã nguồn.

Bảng kiểm kê các thành phần project chính:

| Nhóm | File / thư mục | Nội dung |
|---|---|---|
| Dữ liệu đã xử lý | `data/HomeC_cleaned_final.csv` | Dữ liệu sạch dạng CSV đầy đủ |
| Dữ liệu dùng cho app | `data/HomeC_cleaned_final.zip` | Bản nén được dashboard đọc khi chạy |
| KPI summary | `data/kpi_summary.json` | Tổng điện năng, công suất trung bình, peak load, anomaly count và split rates |
| Thiết bị tiêu thụ cao | `data/top5_appliances.csv` | Bảng xếp hạng thiết bị theo tổng kWh, tỷ trọng, giờ peak và ngày peak |
| Tương quan thời tiết | `data/weather_energy_correlation.csv` | Tương quan giữa biến thời tiết với `use [kW]` và `gen [kW]` |
| Tiền xử lý | `src/HomeC_preprocess.py` | Làm sạch dữ liệu, tạo feature, KPI và visualization |
| Huấn luyện model | `src/train_model.py` | Chạy baseline, so sánh model và export artifact |
| Dashboard | `app/app.py` | Ứng dụng Dash cho KPI, biểu đồ, alert và prediction form |
| Visualization | `visualization/` | 14 biểu đồ tĩnh dùng cho phân tích khám phá |
| Kiến trúc | `report/architecture.png`, `report/architecture.svg` | Sơ đồ kiến trúc/pipeline của project |
| Reproducibility | `requirements.txt`, `README.md` | Thư viện cần cài và hướng dẫn chạy lại project |

## 11. Thảo luận

Kết quả thực nghiệm cho thấy điểm mạnh của dự án không nằm ở việc đạt F1 cao, mà nằm ở việc phát hiện vấn đề phương pháp và đánh giá lại pipeline một cách trung thực.

Ba phát hiện chính:

1. `total_appliance` là một nguồn leakage mạnh vì nó là thành phần cộng của `use [kW]`.
2. Nhãn global 3-sigma trên dữ liệu có mùa vụ dễ biến mùa hè thành anomaly.
3. Sau khi loại leakage, các biến thời tiết và thời gian hiện tại không đủ để dự đoán anomaly với F1 cao.

Baseline một cột `total_appliance` đạt test F1 = 0.4866, cao hơn tất cả cấu hình machine learning. Nếu một câu lệnh threshold đơn giản trên một cột có leakage đã tốt hơn model, thì đóng góp thật của bài không thể là "mô hình dự đoán mạnh". Đóng góp hợp lý hơn là: dự án đã chỉ ra leakage, sửa cách định nghĩa nhãn, bổ sung baseline bắt buộc, và báo cáo trung thực năng lực thật của các đặc trưng còn lại.

Nhãn local 30 ngày là một cải thiện quan trọng vì nó so sánh mỗi điểm dữ liệu với chính bối cảnh gần đây của nó. Tuy nhiên, nhãn tốt hơn không tự động tạo ra thông tin dự đoán tốt hơn. Để dự đoán anomaly hiệu quả, mô hình cần thêm các đặc trưng lịch sử như lagged load, rolling mean, rolling standard deviation, rolling max hoặc trạng thái bật/tắt của thiết bị.

## 12. Hạn chế

Dự án vẫn có một số hạn chế:

1. Nhãn anomaly là proxy thống kê, không phải nhãn sự cố thật được xác minh.
2. Dữ liệu chỉ đến từ một hộ gia đình, nên khả năng tổng quát hóa còn hạn chế.
3. Timeline được tái dựng dựa trên giả định mỗi dòng cách nhau một phút.
4. Ngưỡng local 30 ngày vẫn chưa hấp thụ hết các regime tiêu thụ kéo dài, đặc biệt là tháng 7.
5. Bộ đặc trưng deploy không có thông tin lịch sử tiêu thụ, làm giảm khả năng dự đoán.
6. Ngưỡng quyết định chọn trên validation có thể không ổn định khi áp sang một giai đoạn mùa vụ khác.

## 13. Hướng phát triển

Các hướng cải thiện tiếp theo:

1. Thêm đặc trưng lag của `use [kW]`, ví dụ 5 phút, 30 phút, 1 giờ và 24 giờ trước.
2. Thêm rolling features như rolling mean, rolling std, rolling max và rolling slope.
3. Thêm đặc trưng trạng thái thiết bị, ví dụ thay đổi đột ngột của từng appliance.
4. Dùng rolling-origin validation thay vì chỉ một lần chia 70/15/15.
5. Thử nghiệm trên nhiều hộ gia đình và nhiều năm dữ liệu.
6. Thu thập hoặc gán nhãn các sự kiện anomaly thật thay vì chỉ dùng proxy high-load.
7. Cải thiện calibration để ngưỡng quyết định ổn định hơn giữa validation và test.

## 14. Kết luận

Dự án đã xây dựng được một pipeline hoàn chỉnh cho bài toán Smart Home Energy Anomaly Detection, bao gồm tiền xử lý dữ liệu, visualization, baseline, huấn luyện mô hình, đánh giá kết quả, dashboard và tích hợp giải thích bằng AI.

Kết quả quan trọng nhất là việc phát hiện và xử lý target leakage do `total_appliance`. Baseline một cột `total_appliance` đạt test F1 = 0.4866, cao hơn các mô hình machine learning, chứng minh rằng kết quả ban đầu có thể bị phóng đại bởi leakage. Sau khi loại cột này và dùng nhãn local 30 ngày, hiệu năng F1 của mô hình giảm đáng kể, cho thấy các đặc trưng thời tiết và thời gian hiện tại chưa đủ mạnh cho phát hiện anomaly đáng tin cậy.

Do đó, kết luận trung thực của dự án là: hệ thống hiện tại chưa phải là một anomaly detector đủ tốt để triển khai thực tế, nhưng dự án đã tạo ra một nền tảng thực nghiệm đúng đắn hơn. Dự án nhấn mạnh tầm quan trọng của baseline, kiểm soát leakage, định nghĩa nhãn phù hợp với mùa vụ, chia dữ liệu theo thời gian và báo cáo minh bạch cả khi kết quả mô hình không cao.
