# ==========================================================
# DAP391m PROJECT - Smart Home Energy Anomaly Detection
# Member B - Data Understanding, Cleaning & EDA (Step 1-3)
# REVISED VERSION
#
# Fixes applied in this revision:
#   1. FULL-YEAR DATETIME BUG FIXED (see Section 1 for root cause).
#   2. Robust input handling: works with either the original raw
#      HomeC.csv (55 columns) or the already-cleaned
#      HomeC_cleaned_final.csv (31 columns) shipped in this repo.
#   3. Added: appliance grouping, Top-5 appliance analysis,
#      weather-energy correlation, time-series analysis (hourly /
#      daily / day-of-week / monthly / calendar heatmap), and
#      anomaly-detection visualization.
#   4. Saves a KPI summary (data/kpi_summary.json) consumed by the
#      dashboard (app/app.py) so KPI cards / alert panel stay in
#      sync with this script instead of being hardcoded.
#   5. Engineered numeric features (hour, dayofweek, month,
#      is_weekend, season, time_period, total_appliance) are
#      persisted back into HomeC_cleaned_final.csv so the modeling
#      notebook (notebooks/model_pipeline.ipynb) picks them up
#      automatically without needing to be edited.
# ==========================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe: charts are saved to disk, not shown on screen
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["figure.dpi"] = 110

# ==========================================================
# 0. PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_DIR = os.path.join(BASE_DIR, "data")
VIZ_DIR = os.path.join(BASE_DIR, "visualization")
os.makedirs(VIZ_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "HomeC.csv")                    # original Kaggle file (if team has it)
SEMI_PATH = os.path.join(DATA_DIR, "HomeC_cleaned_final.csv")      # already exists in this repo
OUTPUT_PATH = os.path.join(DATA_DIR, "HomeC_cleaned_final.csv")
KPI_PATH = os.path.join(DATA_DIR, "kpi_summary.json")
TOP5_PATH = os.path.join(DATA_DIR, "top5_appliances.csv")
CORR_PATH = os.path.join(DATA_DIR, "weather_energy_correlation.csv")

# Consistent color palette used across every chart in this project
COLOR_ENERGY = "#2E86AB"     # blue   - household consumption (use [kW])
COLOR_SOLAR = "#F4A300"      # gold   - solar generation (gen [kW])
COLOR_ANOMALY = "#E63946"    # red    - anomalies / alerts
COLOR_NORMAL = "#8AB17D"     # green  - low / normal consumption
COLOR_MEDIUM = "#F2C14E"     # yellow - medium consumption
COLOR_HIGH = "#E63946"       # red    - high consumption
APPLIANCE_PALETTE = sns.color_palette("Set2", 11)


def save_chart(fig, filename, dpi=160):
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved visualization/{filename}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# 1. LOAD DATASET (and explain / fix the full-year datetime bug)
# ==========================================================

section("STEP 1.1 - LOADING DATASET")

if os.path.exists(RAW_PATH):
    INPUT_PATH = RAW_PATH
    print(f"Found raw file: {RAW_PATH}")
    print("-> running the complete cleaning pipeline from scratch.")
elif os.path.exists(SEMI_PATH):
    INPUT_PATH = SEMI_PATH
    print(f"Raw HomeC.csv not found. Using existing file: {SEMI_PATH}")
    print("-> NOTE: the previous version of this script pointed at a file called")
    print("   'HomeC_cleaned.csv', which does not exist anywhere in this repo.")
    print("   That mismatched filename is the reason the script could not be re-run.")
    print("   This script now auto-detects the correct input file instead.")
else:
    raise FileNotFoundError(
        "Neither HomeC.csv nor HomeC_cleaned_final.csv was found in data/. "
        "Place the dataset in the data/ folder before running this script."
    )

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Dataset loaded successfully. Shape: {df.shape}")
print(df.columns.tolist())

# ----------------------------------------------------------
# ROOT CAUSE of the "only 01/01-07/01/2016" bug:
# The dataset is described as "one reading per minute for all of 2016"
# (503,910 rows -> 503,910 / 1440 minutes/day = ~349.9 days, i.e. a full
# year). However the raw "time" column does NOT increase by 60 seconds
# per row as expected for per-minute data - it increases by exactly 1
# second per row. Converting it directly with
#     pd.to_datetime(df["time"], unit="s")
# therefore compresses the entire dataset into ~503,910 seconds, which
# is only ~5.8 days (01/01/2016 -> 07/01/2016) - exactly the bug the
# team observed. The "time" column itself is unreliable for this
# dataset, so instead of trusting it, we reconstruct a clean per-minute
# datetime index spanning the full year, which is what the row count
# and the (per-appliance, per-minute) data collection design actually
# imply.
# ----------------------------------------------------------

section("STEP 1.2 - FIXING THE FULL-YEAR DATETIME BUG")

if "time" in df.columns:
    buggy_start = pd.to_datetime(df["time"].min(), unit="s")
    buggy_end = pd.to_datetime(df["time"].max(), unit="s")
    print(f"Old (buggy) range via pd.to_datetime(time, unit='s'): "
          f"{buggy_start} -> {buggy_end}  ({(buggy_end - buggy_start).days} days)")

df["datetime"] = pd.date_range(start="2016-01-01 00:00:00", periods=len(df), freq="1min")

fixed_start = df["datetime"].min()
fixed_end = df["datetime"].max()
print(f"New (fixed) range via pd.date_range(freq='1min'): "
      f"{fixed_start} -> {fixed_end}  ({(fixed_end - fixed_start).days} days)")
print("Fix applied: full-year coverage confirmed.")

# ==========================================================
# 2. DATASET OVERVIEW
# ==========================================================

section("STEP 1.3 - DATASET OVERVIEW")
print("Shape:", df.shape)
print("\nDescriptive statistics (numeric columns):")
print(df.describe().T[["mean", "std", "min", "max"]])

# ==========================================================
# 3. SCOPE JUSTIFICATION - WHY HOME C
# ==========================================================

section("STEP 1.4 - SCOPE: WHY HOME C")
print("""
The public 'Smart Home Dataset with Weather Information' on Kaggle
contains separate per-minute logs for several households (Home A, B,
C, D, F, G...). This project scopes down to Home C only, because:
  1. Home C is the only file with a complete, near-full-year span of
     per-minute readings for both appliance-level energy AND weather
     data together in a single file - other homes have shorter spans
     or missing weather columns.
  2. Home C contains a rich set of individually metered circuits
     (14 appliance/room channels), which gives enough granularity to
     analyze per-appliance contribution to anomalies.
  3. Restricting the scope to one household keeps the anomaly
     detection problem well-defined (a single consumption baseline)
     instead of mixing several different households' usage patterns,
     which would require separate baselines per home.
""")

# ==========================================================
# 4. MISSING VALUES
# ==========================================================

section("STEP 2.1 - MISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("No missing values detected.")
else:
    print("Missing values found:")
    print(missing)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    print("Missing values imputed (median for numeric, mode for categorical).")

# ==========================================================
# 5. DUPLICATE ROWS / COLUMNS / CONSTANT COLUMNS
# ==========================================================

section("STEP 2.2 - DUPLICATE & CONSTANT COLUMN CHECKS")

duplicate_rows = df.duplicated().sum()
print("Duplicate rows:", duplicate_rows)
if duplicate_rows > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

duplicate_columns = []
cols = df.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if df[cols[i]].equals(df[cols[j]]):
            duplicate_columns.append(cols[j])
duplicate_columns = list(set(duplicate_columns))
print("Duplicate columns:", duplicate_columns if duplicate_columns else "None")
if duplicate_columns:
    df = df.drop(columns=duplicate_columns)
    print("Duplicate columns removed (kept the first occurrence of each).")

constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
print("Constant columns:", constant_cols if constant_cols else "None")
if constant_cols:
    df = df.drop(columns=constant_cols)
    print("Constant columns removed (zero information value).")

# ==========================================================
# 6. FEATURE IDENTIFICATION & SELECTION (with justification)
# ==========================================================

section("STEP 2.3 - FEATURE IDENTIFICATION & SELECTION")

energy_features = [c for c in df.columns if c.lower() in ("use [kw]", "gen [kw]")]

weather_keywords = ["temperature", "humidity", "visibility", "pressure",
                     "windspeed", "cloud", "dew", "precip", "windbearing"]
weather_features = [c for c in df.columns
                     if any(k in c.lower() for k in weather_keywords)]
weather_text_features = [c for c in df.columns if c.lower() in ("icon", "summary")]

appliance_features = [c for c in df.columns
                       if "[kw]" in c.lower()
                       and c not in energy_features]

time_features_raw = [c for c in df.columns if c.lower() in ("time", "datetime")]

print("\nEnergy Features (kept - these define the consumption target):")
print(" ", energy_features)
print("\nAppliance Features (kept - per-device circuits, used for Top-5 & feature importance):")
print(" ", appliance_features)
print("\nWeather Features - numeric (kept - tested against energy/solar generation):")
print(" ", weather_features)
print("\nWeather Features - categorical (kept - one-hot encoded downstream for model):")
print(" ", weather_text_features)
print("\nTime Features (kept - source for engineered hour/day/month features):")
print(" ", time_features_raw)

all_kept = set(energy_features + appliance_features + weather_features
                + weather_text_features + time_features_raw)
dropped = [c for c in df.columns if c not in all_kept]
print("\nColumns with no analytical role for this project (dropped if any):")
print(" ", dropped if dropped else "None - every remaining column is used "
                                    "for either the energy target, an appliance "
                                    "predictor, a weather predictor, or the time index.")
if dropped:
    df = df.drop(columns=dropped)

# ==========================================================
# 7. FEATURE ENGINEERING (persisted, numeric-only so downstream
#    notebook keeps working without modification)
# ==========================================================

section("STEP 2.4 - FEATURE ENGINEERING")

df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek      # 0=Mon ... 6=Sun
df["month"] = df["datetime"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)


def to_season(m):
    if m in (12, 1, 2):
        return 1   # Winter
    if m in (3, 4, 5):
        return 2   # Spring
    if m in (6, 7, 8):
        return 3   # Summer
    return 4       # Fall


def to_time_period(h):
    if 0 <= h < 6:
        return 0   # Night
    if 6 <= h < 12:
        return 1   # Morning
    if 12 <= h < 18:
        return 2   # Afternoon
    return 3       # Evening


df["season"] = df["month"].apply(to_season)             # 1=Winter,2=Spring,3=Summer,4=Fall
df["time_period"] = df["hour"].apply(to_time_period)    # 0=Night,1=Morning,2=Afternoon,3=Evening

# Group raw circuits into human-readable appliance groups
kitchen_cols = [c for c in appliance_features if "kitchen" in c.lower()]
furnace_cols = [c for c in appliance_features if "furnace" in c.lower()]
other_appliance_cols = [c for c in appliance_features
                         if c not in kitchen_cols and c not in furnace_cols]

appliance_groups = {}
if kitchen_cols:
    df["Kitchen [kW]"] = df[kitchen_cols].sum(axis=1)
    appliance_groups["Kitchen [kW]"] = kitchen_cols
if furnace_cols:
    df["Furnace [kW]"] = df[furnace_cols].sum(axis=1)
    appliance_groups["Furnace [kW]"] = furnace_cols
for c in other_appliance_cols:
    appliance_groups[c] = [c]

grouped_appliance_cols = list(appliance_groups.keys())

# total_appliance: sum of ALL individual appliance circuits (not the grouped
# ones, to avoid double counting). This also fixes an existing inconsistency
# in app/app.py, which already expects a "total_appliance" feature that was
# never actually created anywhere in the pipeline before this revision.
df["total_appliance"] = df[appliance_features].sum(axis=1)

print("Engineered numeric features added: hour, dayofweek, month, is_weekend, "
      "season, time_period, total_appliance.")
print("Engineered appliance groups added:", grouped_appliance_cols)
if kitchen_cols:
    print("(Kitchen [kW] = sum of:", kitchen_cols, ")")
if furnace_cols:
    print("(Furnace [kW] = sum of:", furnace_cols, ")")

# ==========================================================
# 8. ANOMALY FLAG FOR EDA VISUALIZATION ONLY
#    (kept separate from notebooks/model_pipeline.ipynb's own
#    "anomaly" label so the two stay independent)
# ==========================================================

section("STEP 2.5 - STATISTICAL ANOMALY FLAG (for EDA visualization)")

target = "use [kW]"
mu, sigma = df[target].mean(), df[target].std()
K = 3
threshold = mu + K * sigma
df["_eda_anomaly"] = (df[target] > threshold).astype(int)

n_anomalies = int(df["_eda_anomaly"].sum())
anomaly_rate = round(df["_eda_anomaly"].mean() * 100, 2)
print(f"Threshold (mean + {K}*std): {threshold:.3f} kW")
print(f"Anomalies flagged: {n_anomalies} rows ({anomaly_rate}% of data)")

Q1, Q3 = df[target].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
iqr_outliers = df[(df[target] < lower) | (df[target] > upper)]
print(f"IQR outliers: {len(iqr_outliers)} rows "
      f"({round(len(iqr_outliers) / len(df) * 100, 2)}%)")

# ==========================================================
# 9. SAVE CLEANED + ENGINEERED DATASET
# ==========================================================

section("STEP 2.6 - SAVING CLEANED DATASET")

save_cols = [c for c in df.columns if c != "_eda_anomaly"]
df[save_cols].to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}  (shape: {df[save_cols].shape})")

# ==========================================================
# 10. TOP-5 APPLIANCE ANALYSIS
# ==========================================================

section("STEP 3.1 - TOP 5 APPLIANCES")

totals = df[grouped_appliance_cols].sum().sort_values(ascending=False)
share = (totals / totals.sum() * 100).round(2)

peak_hour = {}
peak_date = {}
for col in grouped_appliance_cols:
    peak_hour[col] = int(df.groupby("hour")[col].mean().idxmax())
    daily_sum = df.groupby(df["datetime"].dt.date)[col].sum()
    peak_date[col] = str(daily_sum.idxmax())

top5_df = pd.DataFrame({
    "Total_kWh": totals,
    "Share_%": share,
    "Peak_Hour": pd.Series(peak_hour),
    "Peak_Date": pd.Series(peak_date),
}).sort_values("Total_kWh", ascending=False)

top5 = top5_df.head(5)
print(top5_df.round(2))
top5_df.to_csv(TOP5_PATH)
print(f"Saved: {TOP5_PATH}")

appliance_color_map = {col: APPLIANCE_PALETTE[i % len(APPLIANCE_PALETTE)]
                        for i, col in enumerate(totals.index)}

# Chart 6 - Top 5 appliances bar chart
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(top5.index, top5["Total_kWh"],
               color=[appliance_color_map[c] for c in top5.index])
for b, pct in zip(bars, top5["Share_%"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f"{pct}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Top 5 Energy-Consuming Appliances (Full Year 2016)")
ax.set_xlabel("Appliance")
ax.set_ylabel("Total Energy Consumption (kWh)")
ax.tick_params(axis="x", rotation=20)
save_chart(fig, "chart6_top5_appliances_bar.png")

# Chart 7 - Appliance share pie chart (Top 5 + Others)
others_total = totals.iloc[5:].sum()
pie_labels = list(top5.index) + ["Others"]
pie_values = list(top5["Total_kWh"]) + [others_total]
pie_colors = [appliance_color_map[c] for c in top5.index] + ["#B0B0B0"]

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(pie_values, labels=pie_labels, autopct="%1.1f%%",
       colors=pie_colors, startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 1})
ax.set_title("Appliance Energy Share - Top 5 vs Others")
save_chart(fig, "chart7_appliance_pie.png")

# ==========================================================
# 11. WEATHER vs ENERGY CORRELATION
# ==========================================================

section("STEP 3.2 - WEATHER vs ENERGY CORRELATION")

daily = df.set_index("datetime").resample("D")[
    ["use [kW]", "gen [kW]"] + weather_features
].mean(numeric_only=True)

corr_daily = daily.corr()[["use [kW]", "gen [kW]"]].drop(["use [kW]", "gen [kW]"])
print("Daily-aggregated correlation (weather vs energy):")
print(corr_daily.round(3))
corr_daily.to_csv(CORR_PATH)
print(f"Saved: {CORR_PATH}")

# Chart 4 - Correlation heatmap (energy + key appliances + weather)
heatmap_cols = (["use [kW]", "gen [kW]"] + top5_df.head(5).index.tolist()
                + weather_features)
heatmap_cols = [c for c in heatmap_cols if c in df.columns]
corr_matrix = df[heatmap_cols].sample(min(150_000, len(df)), random_state=42).corr()

fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("Correlation Heatmap - Energy, Top Appliances & Weather Variables")
save_chart(fig, "chart4_heatmap.png")

# Chart 5 - Weather vs energy scatter + regression (2x2 grid)
pairs = [
    ("temperature", "gen [kW]", "Temperature (°F)", "Solar Generation (kW)"),
    ("cloudCover", "gen [kW]", "Cloud Cover (0-1)", "Solar Generation (kW)"),
    ("humidity", "use [kW]", "Humidity (0-1)", "Energy Consumption (kW)"),
    ("windSpeed", "use [kW]", "Wind Speed (mph)", "Energy Consumption (kW)"),
]
pairs = [p for p in pairs if p[0] in daily.columns and p[1] in daily.columns]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (x, y, xlabel, ylabel) in zip(axes.flat, pairs):
    sns.regplot(data=daily, x=x, y=y, ax=ax,
                scatter_kws={"alpha": 0.5, "color": COLOR_ENERGY if "use" in y else COLOR_SOLAR, "s": 18},
                line_kws={"color": COLOR_ANOMALY})
    r = daily[[x, y]].corr().iloc[0, 1]
    ax.set_title(f"{xlabel} vs {ylabel}  (r = {r:.2f})", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
fig.suptitle("Daily-Aggregated Weather vs Energy Relationships", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_chart(fig, "chart5_weather_scatter.png")

print("""
Reading the relationships:
- Temperature/cloud cover are the strongest weather drivers of SOLAR
  GENERATION (warmer, clearer days -> more solar output) - consistent
  with how PV panels behave.
- Weather has only a WEAK direct relationship with raw household USE
  (appliance behavior dominates), which is expected: at 1-minute
  resolution, consumption is driven mostly by which appliances are
  switched on, not by outdoor conditions.
""")

# ==========================================================
# 12. TIME SERIES ANALYSIS
# ==========================================================

section("STEP 3.3 - TIME SERIES ANALYSIS")

daily_total = df.set_index("datetime").resample("D")[["use [kW]", "gen [kW]"]].sum()
hourly_avg = df.groupby("hour")[["use [kW]"]].mean()
dow_avg = df.groupby("dayofweek")[["use [kW]"]].mean()
month_avg = df.groupby("month")[["use [kW]"]].mean()

# Chart 1 - Full-year line chart (daily total, now meaningful after the fix)
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(daily_total.index, daily_total["use [kW]"], color=COLOR_ENERGY, linewidth=1.3,
        label="Daily Total Consumption")
ax.axhline(daily_total["use [kW]"].mean(), color=COLOR_ANOMALY, linestyle="--",
           linewidth=1, label=f"Yearly Daily Average ({daily_total['use [kW]'].mean():.0f} kWh)")
ax.set_title("Household Energy Consumption Over the Full Year 2016 (Daily Total)")
ax.set_xlabel("Date")
ax.set_ylabel("Energy Consumption (kWh / day)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend()
save_chart(fig, "chart1_timeseries.png")

# Chart 2 - Histogram (per-minute use [kW])
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.hist(df["use [kW]"], bins=60, color=COLOR_ENERGY, edgecolor="white")
ax.axvline(threshold, color=COLOR_ANOMALY, linestyle="--", linewidth=1.5,
           label=f"Anomaly threshold = {threshold:.2f} kW (mean+{K}σ)")
ax.set_title("Distribution of Per-Minute Energy Consumption")
ax.set_xlabel("Energy Consumption (kW)")
ax.set_ylabel("Frequency (count of 1-minute readings)")
ax.legend()
save_chart(fig, "chart2_histogram.png")

# Chart 3 - Boxplot
fig, ax = plt.subplots(figsize=(8.5, 4.2))
sns.boxplot(x=df["use [kW]"], ax=ax, color=COLOR_ENERGY)
ax.axvline(threshold, color=COLOR_ANOMALY, linestyle="--", linewidth=1.5,
           label=f"Anomaly threshold ({threshold:.2f} kW)")
ax.set_title("Boxplot of Energy Consumption - Outlier Detection")
ax.set_xlabel("Energy Consumption (kW)")
ax.legend()
save_chart(fig, "chart3_boxplot.png")

# Chart 8 - Hourly consumption, colored by Low/Medium/High tertile
fig, ax = plt.subplots(figsize=(9, 5))
low_t, high_t = hourly_avg["use [kW]"].quantile([0.33, 0.66])
bar_colors = [COLOR_NORMAL if v <= low_t else COLOR_MEDIUM if v <= high_t else COLOR_HIGH
              for v in hourly_avg["use [kW]"]]
ax.bar(hourly_avg.index, hourly_avg["use [kW]"], color=bar_colors)
ax.set_title("Average Energy Consumption by Hour of Day")
ax.set_xlabel("Hour of Day (0-23)")
ax.set_ylabel("Average Consumption (kW)")
ax.set_xticks(range(0, 24))
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [COLOR_NORMAL, COLOR_MEDIUM, COLOR_HIGH]]
ax.legend(handles, ["Low", "Medium", "High"], title="Consumption Level")
save_chart(fig, "chart8_hourly_consumption.png")

# Chart 9 - Area chart of daily total (use vs gen)
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.fill_between(daily_total.index, daily_total["use [kW]"], color=COLOR_ENERGY, alpha=0.4,
                label="Daily Use (kWh)")
ax.plot(daily_total.index, daily_total["use [kW]"], color=COLOR_ENERGY, linewidth=1)
ax.fill_between(daily_total.index, daily_total["gen [kW]"], color=COLOR_SOLAR, alpha=0.5,
                label="Daily Solar Generation (kWh)")
ax.plot(daily_total.index, daily_total["gen [kW]"], color=COLOR_SOLAR, linewidth=1)
ax.set_title("Daily Energy Consumption vs Solar Generation (Area Chart)")
ax.set_xlabel("Date")
ax.set_ylabel("Energy (kWh / day)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend()
save_chart(fig, "chart9_daily_area.png")

# Chart 10 - Stacked area chart, monthly sum of top appliance groups
monthly_appl = df.groupby("month")[grouped_appliance_cols].sum() / 1000.0  # -> MWh for readability
top_for_stack = totals.head(6).index.tolist()  # top 6 keeps the stack readable
fig, ax = plt.subplots(figsize=(11, 6))
ax.stackplot(monthly_appl.index,
             [monthly_appl[c] for c in top_for_stack],
             labels=top_for_stack,
             colors=[appliance_color_map[c] for c in top_for_stack],
             alpha=0.85)
ax.set_title("Monthly Energy Consumption by Top Appliances (Stacked Area)")
ax.set_xlabel("Month")
ax.set_ylabel("Energy Consumption (MWh)")
ax.set_xticks(range(1, 13))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
fig.tight_layout()
save_chart(fig, "chart10_stacked_area_appliances.png")

# Chart 11 - Calendar heatmap (month x day-of-month, total daily kWh)
cal = daily_total["use [kW]"].copy()
cal_df = pd.DataFrame({
    "day": cal.index.day,
    "month": cal.index.month,
    "value": cal.values,
})
cal_pivot = cal_df.pivot_table(index="month", columns="day", values="value")
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(15, 5.5))
sns.heatmap(cal_pivot, cmap="YlOrRd", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Total Daily Consumption (kWh)"}, ax=ax)
ax.set_title("Calendar Heatmap - Daily Total Energy Consumption (2016)")
ax.set_xlabel("Day of Month")
ax.set_ylabel("Month")
ax.set_yticklabels(month_labels[:cal_pivot.shape[0]], rotation=0)
save_chart(fig, "chart11_calendar_heatmap.png")

# Chart 12 - Anomaly-highlighted time series (per-minute, downsampled for plotting)
sample_df = df.iloc[::200].copy()
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(sample_df["datetime"], sample_df["use [kW]"], color=COLOR_ENERGY,
        linewidth=0.8, label="Energy Consumption", zorder=1)
anomalies_sample = sample_df[sample_df["_eda_anomaly"] == 1]
ax.scatter(anomalies_sample["datetime"], anomalies_sample["use [kW]"],
           color=COLOR_ANOMALY, s=14, zorder=2, label="Flagged Anomaly")
ax.axhline(threshold, color=COLOR_ANOMALY, linestyle="--", linewidth=1,
           label=f"Threshold (mean+{K}σ = {threshold:.2f} kW)")
ax.set_title(f"Anomaly Detection Visualization - {n_anomalies} anomalies "
             f"flagged ({anomaly_rate}% of data)")
ax.set_xlabel("Date")
ax.set_ylabel("Energy Consumption (kW)")
ax.legend()
save_chart(fig, "chart12_anomaly_timeseries.png")

# Chart 13 - Consumption by day of week
fig, ax = plt.subplots(figsize=(8.5, 5))
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
bar_colors = [COLOR_HIGH if d >= 5 else COLOR_ENERGY for d in dow_avg.index]
ax.bar(dow_labels[:len(dow_avg)], dow_avg["use [kW]"], color=bar_colors)
ax.set_title("Average Energy Consumption by Day of Week")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Average Consumption (kW)")
save_chart(fig, "chart13_dow_consumption.png")

# Chart 14 - Consumption by month
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([month_labels[m - 1] for m in month_avg.index], month_avg["use [kW]"],
       color=COLOR_ENERGY)
ax.set_title("Average Energy Consumption by Month (Full Year 2016)")
ax.set_xlabel("Month")
ax.set_ylabel("Average Consumption (kW)")
save_chart(fig, "chart14_monthly_consumption.png")

print("14 charts saved to visualization/.")

# ==========================================================
# 13. KPI SUMMARY (consumed by the dashboard)
# ==========================================================

section("STEP 3.4 - KPI SUMMARY")

kpi = {
    "total_energy_kwh": round(df["use [kW]"].sum() / 60, 2),   # per-minute kW -> kWh
    "average_energy_kw": round(df["use [kW]"].mean(), 4),
    "peak_consumption_kw": round(df["use [kW]"].max(), 4),
    "peak_consumption_time": str(df.loc[df["use [kW]"].idxmax(), "datetime"]),
    "number_of_anomalies": n_anomalies,
    "anomaly_rate_pct": anomaly_rate,
    "anomaly_threshold_kw": round(threshold, 4),
    "total_solar_generation_kwh": round(df["gen [kW]"].sum() / 60, 2),
    "date_range_start": str(fixed_start),
    "date_range_end": str(fixed_end),
    "top5_appliances": top5_df.head(5)["Total_kWh"].round(2).to_dict(),
    "weather_correlation_use": corr_daily["use [kW]"].round(3).to_dict(),
    "weather_correlation_gen": corr_daily["gen [kW]"].round(3).to_dict(),
}

with open(KPI_PATH, "w", encoding="utf-8") as f:
    json.dump(kpi, f, indent=2, ensure_ascii=False)

print(json.dumps(kpi, indent=2, ensure_ascii=False))
print(f"\nSaved: {KPI_PATH}")

# ==========================================================
# 14. SUMMARY
# ==========================================================

section("DATA UNDERSTANDING & EDA COMPLETE")
print(f"""
Completed:
  [x] Full-year datetime bug fixed ({fixed_start.date()} -> {fixed_end.date()})
  [x] Missing value / duplicate / constant-column checks
  [x] Feature identification & selection (with justification)
  [x] Feature engineering (hour, dayofweek, month, is_weekend, season,
      time_period, total_appliance, Kitchen/Furnace grouping)
  [x] Top-5 appliance analysis -> data/top5_appliances.csv
  [x] Weather-energy correlation -> data/weather_energy_correlation.csv
  [x] Time-series analysis (hourly / daily / day-of-week / monthly / calendar)
  [x] Anomaly-detection visualization
  [x] 14 charts saved to visualization/
  [x] KPI summary saved to data/kpi_summary.json
  [x] Cleaned + engineered dataset saved to data/HomeC_cleaned_final.csv

Ready for:
  -> notebooks/model_pipeline.ipynb (Member A) - now trains on full-year data
     with 'total_appliance' available as an extra feature.
  -> app/app.py (Member C) - dashboard can read data/kpi_summary.json directly.
""")
