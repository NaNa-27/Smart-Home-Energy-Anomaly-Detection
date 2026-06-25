# ==========================================================
# DAP391m - Smart Home Anomaly Detection Dashboard
# Member C: Dashboard + App + Integrating AI Services
#
# Vị trí file: <repo>/app/app.py
# Đọc artifact trực tiếp từ <repo>/notebooks/  (không copy trùng):
#     best_model.pkl, feature_columns.pkl
# Đọc dữ liệu vẽ biểu đồ từ <repo>/data/HomeC_cleaned_final.csv
#     (nhớ giải nén HomeC_cleaned_final.rar trước khi chạy).
#
# Đường dẫn tính theo __file__ -> chạy được dù gọi từ thư mục nào,
# tránh FileNotFoundError như lỗi nhóm từng gặp.
#
# Sửa so với bản cũ của C (capnhat_app.py):
#   1. PREDICT ĐÚNG: nạp feature_columns.pkl (55 cột), dựng đủ cột theo
#      đúng thứ tự train; bản cũ chỉ đưa 5 giá trị -> sai/crash.
#   2. Dash 3.0: app.run() thay cho app.run_server() (đã deprecated).
#   3. AI HYBRID: có GEMINI_API_KEY -> gọi thật; không có -> fallback rule-based.
# ==========================================================

import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib

# ---------- 0. Đường dẫn theo vị trí file (app/ -> repo gốc) ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)
NOTEBOOKS_DIR = os.path.join(REPO_DIR, "notebooks")
DATA_DIR = os.path.join(REPO_DIR, "data")

MODEL_PATH = os.path.join(NOTEBOOKS_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(NOTEBOOKS_DIR, "feature_columns.pkl")
DATA_PATH = os.path.join(DATA_DIR, "HomeC_cleaned_final.csv")

# ---------- 1. Khởi tạo ----------
app = dash.Dash(__name__)
app.title = "DAP391m - Smart Home Anomaly Detection"

# ---------- 2. Dữ liệu mẫu để vẽ biểu đồ ----------
try:
    df = pd.read_csv(DATA_PATH).sample(n=10000, random_state=42)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    data_status = ""
except FileNotFoundError:
    df = pd.DataFrame({
        "datetime": pd.date_range("2016-01-01", periods=10, freq="h"),
        "use [kW]": [1] * 10, "temperature": [70] * 10,
    })
    data_status = (f"Chưa thấy {DATA_PATH} — đang vẽ dữ liệu giả. "
                   f"Hãy giải nén data/HomeC_cleaned_final.rar trước.")

# ---------- 3. Nạp model + DANH SÁCH CỘT (mấu chốt để predict đúng) ----------
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    model_status = f"Mô hình sẵn sàng ({len(feature_columns)} cột feature)."
except FileNotFoundError:
    model = None
    feature_columns = None
    model_status = ("Chưa thấy best_model.pkl / feature_columns.pkl trong notebooks/ "
                    "— đang dùng dự đoán giả lập. Hãy chạy notebook để xuất artifact.")

# ---------- 4. Tích hợp AI: HYBRID ----------
# Đặt biến môi trường trước khi chạy:  export GEMINI_API_KEY="..."
# KHÔNG ghi key cứng trong code (tránh lộ key khi đẩy lên GitHub).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def _ai_fallback(prediction, temp_val, appliance_val):
    """Giải thích rule-based — chạy khi không có API key. Demo không bao giờ chết."""
    if prediction == 1:
        return (f"[AI - chế độ ngoại tuyến] Phát hiện mức tiêu thụ BẤT THƯỜNG. "
                f"Nhiệt độ {temp_val} độ, tổng thiết bị {appliance_val} kW. "
                f"Gợi ý kiểm tra thiết bị sưởi/làm mát hoặc khả năng rò rỉ điện.")
    return (f"[AI - chế độ ngoại tuyến] Mức tiêu thụ BÌNH THƯỜNG với nhiệt độ "
            f"{temp_val} độ, tổng thiết bị {appliance_val} kW. Không có dấu hiệu bất thường.")


def get_ai_explanation(prediction, temp_val, appliance_val):
    """Gọi Gemini API nếu có key; lỗi/không key -> fallback rule-based."""
    if not GEMINI_API_KEY:
        return _ai_fallback(prediction, temp_val, appliance_val)

    label = "BẤT THƯỜNG" if prediction == 1 else "BÌNH THƯỜNG"
    prompt = (
        "Bạn là trợ lý phân tích năng lượng nhà thông minh. "
        f"Mô hình vừa kết luận mức tiêu thụ điện là {label}. "
        f"Bối cảnh: nhiệt độ {temp_val} độ, tổng thiết bị bật {appliance_val} kW. "
        "Giải thích ngắn gọn (3-4 câu) bằng tiếng Việt vì sao có thể như vậy "
        "và đề xuất hành động cho người dùng."
    )
    try:
        import requests
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY)
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return "[AI - Gemini] " + text.strip()
    except Exception as e:
        # Mất mạng / hết quota / sai key -> không làm sập app, quay về fallback
        return _ai_fallback(prediction, temp_val, appliance_val) + f"  (Lý do fallback: {e})"


# ---------- 5. Biểu đồ (RQ1, RQ3) ----------
fig_timeseries = px.line(df, x="datetime", y="use [kW]",
                         title="Tiêu thụ điện theo thời gian")
if "temperature" in df.columns:
    fig_scatter = px.scatter(df, x="temperature", y="use [kW]",
                             title="Tương quan nhiệt độ và tiêu thụ")
else:
    fig_scatter = px.scatter(title="Thiếu cột temperature trong dữ liệu mẫu")

# ---------- 6. Hàm dự đoán đúng chuẩn ----------
def predict_anomaly(user_inputs: dict):
    """
    user_inputs: dict các giá trị người dùng nhập, vd {"temperature": 75, "total_appliance": 2}.
    Dựng 1 hàng đủ TẤT CẢ cột theo feature_columns (cột không nhập = 0),
    đúng thứ tự lúc train -> tránh lệch cột / sai số chiều.
    """
    if model is None or feature_columns is None:
        temp = user_inputs.get("temperature", 0)
        appl = user_inputs.get("total_appliance", 0)
        return 1 if (temp > 80 or appl > 5) else 0

    row = pd.DataFrame([user_inputs]).reindex(columns=feature_columns, fill_value=0)
    return int(model.predict(row)[0])


# ---------- 7. Giao diện ----------
app.layout = html.Div([
    html.H1("DAP391m - Smart Home Anomaly Detection Dashboard",
            style={"textAlign": "center"}),
    html.P(model_status, style={"textAlign": "center",
                                "color": "green" if model is not None else "darkorange"}),
    html.P(data_status, style={"textAlign": "center", "color": "darkorange"})
    if data_status else html.Div(),

    html.Div([
        html.H3("Kiểm tra bất thường và AI phân tích"),
        html.Label("Nhiệt độ (temperature): "),
        dcc.Input(id="input-temp", type="number", value=75, step=1),
        html.Label("  Tổng thiết bị đang bật (total_appliance, kW): "),
        dcc.Input(id="input-appliance", type="number", value=2, step=0.1),
        html.Br(), html.Br(),
        html.Button("Dự đoán và hỏi AI", id="predict-btn", n_clicks=0,
                    style={"fontSize": "16px", "padding": "10px"}),
        html.Div(id="prediction-output",
                 style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "20px"}),
        html.Div(id="ai-output",
                 style={"marginTop": "10px", "padding": "10px",
                        "backgroundColor": "#e8f4f8", "borderRadius": "5px"}),
    ], style={"padding": "20px", "backgroundColor": "#f0f0f0",
              "marginBottom": "20px", "borderRadius": "10px"}),

    html.Div([
        html.Div([dcc.Graph(figure=fig_timeseries)],
                 style={"width": "50%", "display": "inline-block"}),
        html.Div([dcc.Graph(figure=fig_scatter)],
                 style={"width": "50%", "display": "inline-block"}),
    ]),
])


# ---------- 8. Callback ----------
@app.callback(
    [Output("prediction-output", "children"),
     Output("ai-output", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("input-temp", "value"),
     State("input-appliance", "value")],
)
def update_prediction(n_clicks, temp, appliance):
    if not n_clicks:
        return "", ""

    # Map ô nhập -> tên cột thật của dataset. Các cột khác để mặc định 0.
    user_inputs = {"temperature": temp, "total_appliance": appliance}
    pred = predict_anomaly(user_inputs)

    text_result = "KẾT QUẢ: BẤT THƯỜNG" if pred == 1 else "KẾT QUẢ: BÌNH THƯỜNG"
    ai_text = get_ai_explanation(pred, temp, appliance)
    return text_result, ai_text


# ---------- 9. Chạy app (Dash 3.0: dùng app.run, KHÔNG app.run_server) ----------
if __name__ == "__main__":
    app.run(debug=True)
