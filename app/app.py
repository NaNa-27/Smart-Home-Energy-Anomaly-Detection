# ==========================================================
# DAP391m - Smart Home Anomaly Detection Dashboard
# Member C: Dashboard + App + Integrating AI Services
# REVISED VERSION - full KPI / chart / alert dashboard
#
# Vị trí file: <repo>/app/app.py
# Đọc artifact trực tiếp từ <repo>/notebooks/  (không copy trùng):
#     best_model.pkl, feature_columns.pkl
# Đọc dữ liệu vẽ biểu đồ từ <repo>/data/HomeC_cleaned_final.csv
#     (nhớ giải nén HomeC_cleaned_final.rar trước khi chạy, hoặc chạy
#     src/HomeC_preprocess.py trước để sinh bản mới nhất).
# Đọc data/kpi_summary.json (do src/HomeC_preprocess.py sinh ra) để
#     KPI Cards & Alert Panel luôn khớp với số liệu EDA, không hardcode.
#
# Đường dẫn tính theo __file__ -> chạy được dù gọi từ thư mục nào.
#
# Thay đổi so với bản trước:
#   1. PREDICT ĐÚNG: nạp feature_columns.pkl, dựng đủ cột theo đúng
#      thứ tự train (giữ nguyên từ bản trước).
#   2. Dash 3.x: dùng app.run() (không dùng app.run_server() đã deprecated).
#   3. AI HYBRID: có GEMINI_API_KEY -> gọi thật; không có -> fallback rule-based.
#   4. MỚI: KPI Cards (Total/Average/Peak/Anomalies), Alert Panel,
#      Smart City Recommendation panel, và 10 biểu đồ Plotly tương tác
#      (Line, Area, Stacked Area, Correlation Heatmap, Weather Scatter +
#      Regression, Top-5 Bar, Pie, Hourly Bar, Day-of-week Bar, Calendar
#      Heatmap) - tất cả dùng cùng bảng màu với visualization/*.png để
#      đồng bộ giữa báo cáo (ảnh tĩnh) và dashboard (tương tác).
# ==========================================================

import os
import json

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------- 0. Đường dẫn theo vị trí file (app/ -> repo gốc) ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)
NOTEBOOKS_DIR = os.path.join(REPO_DIR, "notebooks")
DATA_DIR = os.path.join(REPO_DIR, "data")

MODEL_PATH = os.path.join(NOTEBOOKS_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(NOTEBOOKS_DIR, "feature_columns.pkl")
DATA_PATH = os.path.join(DATA_DIR, "HomeC_cleaned_final.csv")
KPI_PATH = os.path.join(DATA_DIR, "kpi_summary.json")
TOP5_PATH = os.path.join(DATA_DIR, "top5_appliances.csv")
CORR_PATH = os.path.join(DATA_DIR, "weather_energy_correlation.csv")

# Consistent color palette - SAME hex values as src/HomeC_preprocess.py
# so the static report charts and this interactive dashboard always agree.
COLOR_ENERGY = "#2E86AB"
COLOR_SOLAR = "#F4A300"
COLOR_ANOMALY = "#E63946"
COLOR_NORMAL = "#8AB17D"
COLOR_MEDIUM = "#F2C14E"
COLOR_HIGH = "#E63946"
APPLIANCE_COLORWAY = px.colors.qualitative.Set2

# ---------- 1. Khởi tạo ----------
app = dash.Dash(__name__)
app.title = "DAP391m - Smart Home Anomaly Detection"

# ---------- 2. Dữ liệu cho dashboard ----------
data_status = ""
weather_features_guess = ["temperature", "apparentTemperature", "humidity", "visibility",
                           "pressure", "windSpeed", "cloudCover", "windBearing",
                           "precipIntensity", "dewPoint", "precipProbability"]

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])

    appliance_cols_guess = [c for c in df.columns if "[kW]" in c and c not in
                             ("use [kW]", "gen [kW]")]
    # prefer the grouped columns already produced by HomeC_preprocess.py
    grouped_cols = [c for c in ["Kitchen [kW]", "Furnace [kW]", "Dishwasher [kW]",
                                 "Home office [kW]", "Fridge [kW]", "Wine cellar [kW]",
                                 "Garage door [kW]", "Barn [kW]", "Well [kW]",
                                 "Microwave [kW]", "Living room [kW]"] if c in df.columns]
    if not grouped_cols:
        grouped_cols = appliance_cols_guess

    weather_cols = [c for c in weather_features_guess if c in df.columns]

    daily_total = df.set_index("datetime").resample("D")[["use [kW]", "gen [kW]"]].sum()
    hourly_avg = df.groupby("hour")[["use [kW]"]].mean() if "hour" in df.columns else \
        df.assign(hour=df["datetime"].dt.hour).groupby("hour")[["use [kW]"]].mean()
    dow_avg = df.groupby("dayofweek")[["use [kW]"]].mean() if "dayofweek" in df.columns else \
        df.assign(dayofweek=df["datetime"].dt.dayofweek).groupby("dayofweek")[["use [kW]"]].mean()
    monthly_appl = (df.groupby("month")[grouped_cols].sum() / 1000.0) if "month" in df.columns \
        else pd.DataFrame()

    daily_weather = df.set_index("datetime").resample("D")[
        ["use [kW]", "gen [kW]"] + weather_cols
    ].mean(numeric_only=True)

    mu, sigma = df["use [kW]"].mean(), df["use [kW]"].std()
    threshold = mu + 3 * sigma
    anomalies = df[df["use [kW]"] > threshold].copy()
    if grouped_cols:
        anomalies["top_appliance"] = df.loc[anomalies.index, grouped_cols].idxmax(axis=1)

    data_status = ""
except FileNotFoundError:
    df = pd.DataFrame({
        "datetime": pd.date_range("2016-01-01", periods=10, freq="h"),
        "use [kW]": [1] * 10, "gen [kW]": [0.1] * 10, "temperature": [70] * 10,
        "hour": list(range(10)), "dayofweek": [0] * 10, "month": [1] * 10,
    })
    grouped_cols, weather_cols = [], []
    daily_total = df.set_index("datetime")[["use [kW]", "gen [kW]"]]
    hourly_avg = df.groupby("hour")[["use [kW]"]].mean()
    dow_avg = df.groupby("dayofweek")[["use [kW]"]].mean()
    monthly_appl = pd.DataFrame()
    daily_weather = daily_total.copy()
    threshold = 5.0
    anomalies = df.iloc[0:0]
    data_status = (f"Chưa thấy {DATA_PATH} — đang vẽ dữ liệu giả. "
                   f"Hãy chạy src/HomeC_preprocess.py hoặc giải nén "
                   f"data/HomeC_cleaned_final.rar trước.")

# ---------- 2b. KPI summary (ưu tiên đọc từ kpi_summary.json) ----------
try:
    with open(KPI_PATH, "r", encoding="utf-8") as f:
        kpi = json.load(f)
except FileNotFoundError:
    kpi = {
        "total_energy_kwh": round(df["use [kW]"].sum() / 60, 2),
        "average_energy_kw": round(df["use [kW]"].mean(), 4),
        "peak_consumption_kw": round(df["use [kW]"].max(), 4),
        "peak_consumption_time": str(df.loc[df["use [kW]"].idxmax(), "datetime"]),
        "number_of_anomalies": int(len(anomalies)),
        "anomaly_rate_pct": round(len(anomalies) / max(len(df), 1) * 100, 2),
        "anomaly_threshold_kw": round(threshold, 4),
    }

# ---------- 2c. Top-5 appliances (ưu tiên đọc từ top5_appliances.csv) ----------
try:
    top5_df = pd.read_csv(TOP5_PATH, index_col=0).sort_values("Total_kWh", ascending=False)
except FileNotFoundError:
    if grouped_cols:
        totals = df[grouped_cols].sum().sort_values(ascending=False)
        top5_df = pd.DataFrame({"Total_kWh": totals, "Share_%": (totals / totals.sum() * 100).round(2)})
    else:
        top5_df = pd.DataFrame({"Total_kWh": [], "Share_%": []})

top5 = top5_df.head(5)
appliance_color_map = {col: APPLIANCE_COLORWAY[i % len(APPLIANCE_COLORWAY)]
                        for i, col in enumerate(top5_df.index)}

# ---------- 2d. Weather-energy correlation (ưu tiên đọc từ csv có sẵn) ----------
try:
    corr_daily = pd.read_csv(CORR_PATH, index_col=0)
except FileNotFoundError:
    if weather_cols:
        corr_daily = daily_weather.corr()[["use [kW]", "gen [kW]"]].drop(
            ["use [kW]", "gen [kW]"], errors="ignore")
    else:
        corr_daily = pd.DataFrame()

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    print(f"[AI] Đã nhận GEMINI_API_KEY (đuôi ...{GEMINI_API_KEY[-4:]}). Chế độ: Gemini thật.")
else:
    print("[AI] KHÔNG thấy GEMINI_API_KEY. Chế độ: fallback rule-based.")


# ---------- 4. Tích hợp AI: HYBRID ----------
def _ai_fallback(prediction, temp_val, appliance_val):
    if prediction == 1:
        return (f"[AI - chế độ ngoại tuyến] Phát hiện mức tiêu thụ BẤT THƯỜNG. "
                f"Nhiệt độ {temp_val} độ, tổng thiết bị {appliance_val} kW. "
                f"Gợi ý kiểm tra thiết bị sưởi/làm mát hoặc khả năng rò rỉ điện.")
    return (f"[AI - chế độ ngoại tuyến] Mức tiêu thụ BÌNH THƯỜNG với nhiệt độ "
            f"{temp_val} độ, tổng thiết bị {appliance_val} kW. Không có dấu hiệu bất thường.")


def get_ai_explanation(prediction, temp_val, appliance_val):
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
               "gemini-2.5-flash-lite:generateContent?key=" + GEMINI_API_KEY)
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return "[AI - Gemini] " + text.strip()
    except Exception as e:
        return _ai_fallback(prediction, temp_val, appliance_val) + f"  (Lý do fallback: {e})"


# ---------- 5. Hàm dự đoán đúng chuẩn ----------
def predict_anomaly(user_inputs: dict):
    if model is None or feature_columns is None:
        temp = user_inputs.get("temperature", 0)
        appl = user_inputs.get("total_appliance", 0)
        return 1 if (temp > 80 or appl > 5) else 0

    row = pd.DataFrame([user_inputs]).reindex(columns=feature_columns, fill_value=0)
    return int(model.predict(row)[0])


# ==========================================================
# 6. BIỂU ĐỒ (Plotly) - dùng chung bảng màu với visualization/*.png
# ==========================================================

# 6.1 Line chart - tiêu thụ điện theo ngày, cả năm
fig_line = px.line(daily_total.reset_index(), x="datetime", y="use [kW]",
                    title="Tiêu thụ điện theo ngày (Cả năm 2016)",
                    labels={"datetime": "Ngày", "use [kW]": "Tiêu thụ (kWh/ngày)"})
fig_line.update_traces(line_color=COLOR_ENERGY)
fig_line.add_hline(y=daily_total["use [kW]"].mean(), line_dash="dash", line_color=COLOR_ANOMALY,
                    annotation_text="Trung bình năm")

# 6.2 Area chart - use vs gen
fig_area = go.Figure()
fig_area.add_trace(go.Scatter(x=daily_total.index, y=daily_total["use [kW]"], name="Tiêu thụ (Use)",
                               fill="tozeroy", line_color=COLOR_ENERGY))
fig_area.add_trace(go.Scatter(x=daily_total.index, y=daily_total["gen [kW]"], name="Phát điện mặt trời (Gen)",
                               fill="tozeroy", line_color=COLOR_SOLAR))
fig_area.update_layout(title="Tiêu thụ điện vs Phát điện mặt trời (theo ngày)",
                        xaxis_title="Ngày", yaxis_title="Năng lượng (kWh/ngày)")

# 6.3 Stacked area chart - top appliances theo tháng
fig_stacked = go.Figure()
if not monthly_appl.empty:
    top_for_stack = top5_df.head(6).index.tolist()
    top_for_stack = [c for c in top_for_stack if c in monthly_appl.columns]
    for c in top_for_stack:
        fig_stacked.add_trace(go.Scatter(x=monthly_appl.index, y=monthly_appl[c], name=c,
                                          stackgroup="one",
                                          line_color=appliance_color_map.get(c)))
fig_stacked.update_layout(title="Tiêu thụ điện theo tháng - Top thiết bị (Stacked Area)",
                            xaxis_title="Tháng", yaxis_title="Năng lượng (MWh)")

# 6.4 Correlation heatmap
heatmap_cols = (["use [kW]", "gen [kW]"] + top5_df.head(5).index.tolist() + weather_cols)
heatmap_cols = [c for c in heatmap_cols if c in df.columns]
if len(heatmap_cols) >= 2:
    sample_n = min(150_000, len(df))
    corr_matrix = df[heatmap_cols].sample(sample_n, random_state=42).corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Correlation Heatmap - Energy, Appliances & Weather")
else:
    fig_heatmap = px.imshow([[0]], title="Không đủ cột để vẽ correlation heatmap")

# 6.5 Weather scatter + regression (numpy polyfit, không cần statsmodels)
weather_pairs = [
    ("temperature", "gen [kW]", "Temperature (°F)", "Solar Generation (kW)"),
    ("cloudCover", "gen [kW]", "Cloud Cover (0-1)", "Solar Generation (kW)"),
    ("humidity", "use [kW]", "Humidity (0-1)", "Energy Consumption (kW)"),
    ("windSpeed", "use [kW]", "Wind Speed (mph)", "Energy Consumption (kW)"),
]
weather_pairs = [p for p in weather_pairs if p[0] in daily_weather.columns and p[1] in daily_weather.columns]
default_pair = weather_pairs[0] if weather_pairs else (None, None, None, None)


def build_weather_fig(x_col, y_col, xlabel, ylabel):
    if x_col is None:
        return px.scatter(title="Không có dữ liệu thời tiết phù hợp")
    sub = daily_weather[[x_col, y_col]].dropna()
    fig = px.scatter(sub, x=x_col, y=y_col,
                      color_discrete_sequence=[COLOR_ENERGY if "use" in y_col else COLOR_SOLAR],
                      labels={x_col: xlabel, y_col: ylabel})
    if len(sub) > 2:
        z = np.polyfit(sub[x_col], sub[y_col], 1)
        x_line = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
        y_line = np.polyval(z, x_line)
        r = sub.corr().iloc[0, 1]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                  line_color=COLOR_ANOMALY, name=f"Regression (r={r:.2f})"))
    fig.update_layout(title=f"{xlabel} vs {ylabel} (theo ngày)")
    return fig


fig_weather = build_weather_fig(*default_pair)

# 6.6 Top-5 appliances bar chart
fig_top5 = px.bar(top5.reset_index(), x="index", y="Total_kWh", text="Share_%",
                   color="index", color_discrete_map=appliance_color_map,
                   title="Top 5 thiết bị tiêu thụ điện nhiều nhất (Cả năm 2016)",
                   labels={"index": "Thiết bị", "Total_kWh": "Tổng tiêu thụ (kWh)"})
fig_top5.update_traces(texttemplate="%{text}%", textposition="outside")
fig_top5.update_layout(showlegend=False)

# 6.7 Pie chart - Top5 vs Others
if not top5_df.empty:
    others_total = top5_df["Total_kWh"].iloc[5:].sum()
    pie_labels = list(top5.index) + (["Others"] if others_total > 0 else [])
    pie_values = list(top5["Total_kWh"]) + ([others_total] if others_total > 0 else [])
    pie_colors = [appliance_color_map.get(c) for c in top5.index] + (["#B0B0B0"] if others_total > 0 else [])
    fig_pie = go.Figure(go.Pie(labels=pie_labels, values=pie_values,
                                marker_colors=pie_colors, hole=0.35))
    fig_pie.update_layout(title="Tỷ trọng tiêu thụ điện theo thiết bị")
else:
    fig_pie = px.pie(title="Không có dữ liệu thiết bị")

# 6.8 Hourly consumption (màu theo mức Low/Medium/High)
low_t, high_t = hourly_avg["use [kW]"].quantile([0.33, 0.66])
hourly_colors = [COLOR_NORMAL if v <= low_t else COLOR_MEDIUM if v <= high_t else COLOR_HIGH
                  for v in hourly_avg["use [kW]"]]
fig_hourly = go.Figure(go.Bar(x=hourly_avg.index, y=hourly_avg["use [kW]"], marker_color=hourly_colors))
fig_hourly.update_layout(title="Tiêu thụ điện trung bình theo giờ trong ngày",
                          xaxis_title="Giờ (0-23)", yaxis_title="Tiêu thụ TB (kW)")

# 6.9 Day-of-week consumption
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(dow_avg)]
dow_colors = [COLOR_HIGH if d >= 5 else COLOR_ENERGY for d in dow_avg.index]
fig_dow = go.Figure(go.Bar(x=dow_labels, y=dow_avg["use [kW]"], marker_color=dow_colors))
fig_dow.update_layout(title="Tiêu thụ điện trung bình theo ngày trong tuần",
                       xaxis_title="Ngày trong tuần", yaxis_title="Tiêu thụ TB (kW)")

# 6.10 Calendar heatmap (month x day-of-month)
cal_df = pd.DataFrame({
    "day": daily_total.index.day,
    "month": daily_total.index.month,
    "value": daily_total["use [kW]"].values,
})
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
if len(cal_df) > 0:
    cal_pivot = cal_df.pivot_table(index="month", columns="day", values="value")
    fig_calendar = px.imshow(cal_pivot, color_continuous_scale="YlOrRd",
                              labels={"x": "Ngày trong tháng", "y": "Tháng", "color": "kWh"},
                              title="Calendar Heatmap - Tổng tiêu thụ điện theo ngày (2016)")
    fig_calendar.update_yaxes(tickmode="array", tickvals=list(cal_pivot.index),
                               ticktext=[month_labels[m - 1] for m in cal_pivot.index])
else:
    fig_calendar = px.imshow([[0]], title="Không đủ dữ liệu cho calendar heatmap")


# ==========================================================
# 7. ALERT PANEL - cảnh báo bất thường (top N theo mức độ vượt ngưỡng)
# ==========================================================

def build_alert_cards(anomalies_df, threshold_val, max_alerts=5):
    if anomalies_df.empty:
        return [html.Div("Không phát hiện bất thường nào trong dữ liệu hiện tại.",
                          style={"color": COLOR_NORMAL, "fontWeight": "bold"})]

    top_alerts = anomalies_df.sort_values("use [kW]", ascending=False).head(max_alerts)
    cards = []
    for _, row in top_alerts.iterrows():
        appliance_note = (f" Thiết bị đóng góp nhiều nhất: {row['top_appliance']}."
                           if "top_appliance" in row and pd.notna(row.get("top_appliance"))
                           else "")
        cards.append(html.Div([
            html.Span("⚠ ", style={"color": COLOR_ANOMALY, "fontSize": "18px"}),
            html.B(f"{row['datetime']}"),
            html.Span(f" — Tiêu thụ {row['use [kW]']:.2f} kW vượt ngưỡng bình thường "
                      f"({threshold_val:.2f} kW).{appliance_note}"),
        ], style={"padding": "8px 12px", "borderLeft": f"4px solid {COLOR_ANOMALY}",
                  "backgroundColor": "#FDEDEE", "marginBottom": "6px", "borderRadius": "4px"}))
    return cards


alert_cards = build_alert_cards(anomalies, threshold)


# ==========================================================
# 8. SMART CITY RECOMMENDATION PANEL (data-driven)
# ==========================================================

def build_recommendations(top5_df, corr_daily):
    recs = []
    if not top5_df.empty:
        top_name = top5_df.index[0]
        top_share = top5_df["Share_%"].iloc[0] if "Share_%" in top5_df.columns else None
        if top_share is not None:
            recs.append(f"{top_name} chiếm {top_share}% tổng tiêu thụ thiết bị — "
                        f"ưu tiên hàng đầu để tiết kiệm điện là điều chỉnh lịch hoạt động "
                        f"của thiết bị này (ví dụ: hẹn giờ, điều chỉnh nhiệt độ cài đặt).")
        if "Peak_Hour" in top5_df.columns:
            ph = top5_df["Peak_Hour"].iloc[0]
            recs.append(f"{top_name} hoạt động mạnh nhất vào khoảng {ph}:00 — "
                        f"có thể dịch một phần hoạt động sang giờ thấp điểm để giảm tải.")

    if isinstance(corr_daily, pd.DataFrame) and "gen [kW]" in corr_daily.columns:
        gen_corr = corr_daily["gen [kW]"].dropna()
        if "temperature" in gen_corr.index:
            recs.append(f"Phát điện mặt trời tương quan với nhiệt độ (r={gen_corr['temperature']:.2f}) "
                        f"— nên tận dụng tối đa các thiết bị tiêu thụ nhiều điện vào những ngày "
                        f"nắng ấm để dùng trực tiếp điện mặt trời, giảm điện lưới.")
        if "cloudCover" in gen_corr.index and gen_corr["cloudCover"] < 0:
            recs.append("Độ che phủ mây càng cao thì phát điện mặt trời càng giảm — "
                        "hệ thống nên tự động chuyển sang cảnh báo 'ưu tiên tiết kiệm' "
                        "vào các ngày nhiều mây.")

    recs.append("Lắp lịch tự động: dịch chuyển Dishwasher/Microwave sang giờ thấp điểm "
               "(ban đêm hoặc giữa trưa khi có nắng) để giảm chi phí điện giờ cao điểm.")
    recs.append("Khi hệ thống phát cảnh báo bất thường liên tục từ một thiết bị, "
               "gửi thông báo Smart City để người dùng kiểm tra thiết bị (nguy cơ hỏng/rò điện).")
    return recs[:6]


recommendations = build_recommendations(top5_df, corr_daily)


# ==========================================================
# 9. KPI CARDS
# ==========================================================

def kpi_card(title, value, subtitle="", color=COLOR_ENERGY):
    return html.Div([
        html.Div(title, style={"fontSize": "13px", "color": "#666", "fontWeight": "600",
                                "textTransform": "uppercase"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "color": color}),
        html.Div(subtitle, style={"fontSize": "12px", "color": "#888"}),
    ], style={"flex": "1", "padding": "16px 20px", "backgroundColor": "white",
              "borderRadius": "10px", "boxShadow": "0 1px 4px rgba(0,0,0,0.12)",
              "margin": "0 8px", "minWidth": "180px"})


kpi_row = html.Div([
    kpi_card("Total Energy", f"{kpi.get('total_energy_kwh', 0):,.0f} kWh",
             f"{kpi.get('date_range_start', '')[:10]} → {kpi.get('date_range_end', '')[:10]}",
             COLOR_ENERGY),
    kpi_card("Average Energy", f"{kpi.get('average_energy_kw', 0):.3f} kW",
             "Trung bình mỗi phút", COLOR_ENERGY),
    kpi_card("Peak Consumption", f"{kpi.get('peak_consumption_kw', 0):.2f} kW",
             f"Lúc {kpi.get('peak_consumption_time', 'N/A')}", COLOR_SOLAR),
    kpi_card("Number of Anomalies", f"{kpi.get('number_of_anomalies', 0):,}",
             f"{kpi.get('anomaly_rate_pct', 0)}% tổng dữ liệu", COLOR_ANOMALY),
], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "20px"})


# ==========================================================
# 10. GIAO DIỆN
# ==========================================================

def chart_card(fig, width="50%"):
    return html.Div([dcc.Graph(figure=fig)],
                     style={"width": width, "display": "inline-block", "padding": "6px",
                            "boxSizing": "border-box"})


weather_options = [{"label": f"{xl} vs {yl}", "value": i} for i, (x, y, xl, yl) in enumerate(weather_pairs)]

app.layout = html.Div([
    html.H1("DAP391m - Smart Home Anomaly Detection Dashboard",
            style={"textAlign": "center"}),
    html.P(model_status, style={"textAlign": "center",
                                "color": "green" if model is not None else "darkorange"}),
    html.P(data_status, style={"textAlign": "center", "color": "darkorange"})
    if data_status else html.Div(),

    # ----- KPI CARDS -----
    html.H3("Chỉ số tổng quan (KPI Cards)", style={"marginTop": "10px"}),
    kpi_row,

    # ----- ALERT PANEL -----
    html.H3("Cảnh báo bất thường (Alert Panel)"),
    html.Div(alert_cards, style={"marginBottom": "20px"}),

    # ----- PREDICTION + AI -----
    html.Div([
        html.H3("Kiểm tra bất thường và AI phân tích"),
        html.Label("Nhiệt độ (temperature, °F): "),
        dcc.Input(id="input-temp", type="number", value=75, step=1),
        html.Label("   Tổng thiết bị đang bật (total_appliance, kW): "),
        dcc.Input(id="input-appliance", type="number", value=2, step=0.1),
        html.Label("   Giờ trong ngày (hour): "),
        dcc.Input(id="input-hour", type="number", value=12, step=1, min=0, max=23),
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

    # ----- CHARTS -----
    html.H3("Biểu đồ phân tích"),
    html.Div([
        chart_card(fig_line),
        chart_card(fig_area),
        chart_card(fig_top5),
        chart_card(fig_pie),
        chart_card(fig_hourly),
        chart_card(fig_dow),
        chart_card(fig_heatmap),
        chart_card(fig_stacked),
        chart_card(fig_calendar, width="100%"),
    ]),

    html.H3("Tương quan Thời tiết - Năng lượng"),
    html.Div([
        html.Label("Chọn cặp biến để xem scatter + regression: "),
        dcc.Dropdown(id="weather-pair-dropdown", options=weather_options,
                     value=0 if weather_options else None, style={"width": "400px"}),
    ], style={"marginBottom": "10px"}),
    dcc.Graph(id="weather-graph", figure=fig_weather),

    # ----- SMART CITY RECOMMENDATIONS -----
    html.H3("Đề xuất tiết kiệm điện (Smart City Recommendation)"),
    html.Ul([html.Li(r) for r in recommendations],
            style={"backgroundColor": "#EAF7EE", "padding": "16px 30px",
                  "borderRadius": "10px", "lineHeight": "1.6"}),
], style={"fontFamily": "Arial, sans-serif", "padding": "20px"})


# ---------- 11. Callbacks ----------
@app.callback(
    [Output("prediction-output", "children"),
     Output("ai-output", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("input-temp", "value"),
     State("input-appliance", "value"),
     State("input-hour", "value")],
)
def update_prediction(n_clicks, temp, appliance, hour):
    if not n_clicks:
        return "", ""

    user_inputs = {"temperature": temp, "total_appliance": appliance, "hour": hour}
    pred = predict_anomaly(user_inputs)

    text_result = "KẾT QUẢ: BẤT THƯỜNG" if pred == 1 else "KẾT QUẢ: BÌNH THƯỜNG"
    ai_text = get_ai_explanation(pred, temp, appliance)
    return text_result, ai_text


@app.callback(
    Output("weather-graph", "figure"),
    Input("weather-pair-dropdown", "value"),
)
def update_weather_graph(idx):
    if idx is None or not weather_pairs:
        return fig_weather
    x, y, xl, yl = weather_pairs[idx]
    return build_weather_fig(x, y, xl, yl)


# ---------- 12. Chạy app (Dash 3.x: dùng app.run, KHÔNG app.run_server) ----------
if __name__ == "__main__":
    app.run(debug=True)
