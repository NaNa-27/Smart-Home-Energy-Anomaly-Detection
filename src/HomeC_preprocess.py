"""Preprocess and visualize the HomeC smart-home energy dataset.

Key guarantees in this revision:
- Reads either data/HomeC.csv, data/HomeC_cleaned_final.csv, or the shipped ZIP.
- Is idempotent: previously engineered columns are removed before rebuilding them.
- Converts one-minute power readings (kW) to energy (kWh) with sum / 60.
- Writes the cleaned dataset back as data/HomeC_cleaned_final.zip to keep the project small.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = BASE_DIR / "visualization"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / "HomeC.csv"
CSV_PATH = DATA_DIR / "HomeC_cleaned_final.csv"
ZIP_PATH = DATA_DIR / "HomeC_cleaned_final.zip"
KPI_PATH = DATA_DIR / "kpi_summary.json"
TOP5_PATH = DATA_DIR / "top5_appliances.csv"
CORR_PATH = DATA_DIR / "weather_energy_correlation.csv"

SAMPLE_INTERVAL_MINUTES = 1
SAMPLE_INTERVAL_HOURS = SAMPLE_INTERVAL_MINUTES / 60.0

COLOR_ENERGY = "#2E86AB"
COLOR_SOLAR = "#F4A300"
COLOR_ANOMALY = "#E63946"
COLOR_NORMAL = "#8AB17D"
COLOR_MEDIUM = "#F2C14E"
COLOR_HIGH = "#E63946"

sns.set_style("whitegrid")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["figure.dpi"] = 110

ENGINEERED_COLUMNS = {
    "hour", "dayofweek", "month", "is_weekend", "season", "time_period",
    "Kitchen [kW]", "Furnace [kW]", "total_appliance", "_eda_anomaly",
}
ENERGY_COLUMNS = {"use [kW]", "gen [kW]"}


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def save_chart(fig: plt.Figure, filename: str, dpi: int = 160) -> None:
    path = VIZ_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved visualization/{filename}")


def read_input() -> tuple[pd.DataFrame, Path]:
    if RAW_PATH.exists():
        source = RAW_PATH
    elif CSV_PATH.exists():
        source = CSV_PATH
    elif ZIP_PATH.exists():
        source = ZIP_PATH
    else:
        raise FileNotFoundError(
            "HomeC.csv, HomeC_cleaned_final.csv, or HomeC_cleaned_final.zip"
            "were not found in the data."
        )
    print(f"Reading: {source}")
    return pd.read_csv(source, low_memory=False), source


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_names: list[str] = []
    columns = list(df.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            if df[left].equals(df[right]):
                duplicate_names.append(right)
    if duplicate_names:
        duplicate_names = sorted(set(duplicate_names))
        print("Duplicate columns removed:", duplicate_names)
        return df.drop(columns=duplicate_names)
    print("Duplicate columns: None")
    return df


def to_season(month: pd.Series) -> pd.Series:
    return month.map({12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
                      6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}).astype("int8")


def to_time_period(hour: pd.Series) -> pd.Series:
    return pd.cut(hour, bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype("int8")


def main() -> None:
    section("STEP 1: LOAD AND NORMALIZE")
    df, source = read_input()
    df.columns = [str(c).strip() for c in df.columns]
    print("Loaded shape:", df.shape)

    # Make reruns deterministic: discard every derived column before rebuilding it.
    existing_engineered = [c for c in ENGINEERED_COLUMNS if c in df.columns]
    if existing_engineered:
        df = df.drop(columns=existing_engineered)
        print("Removed stale engineered columns:", sorted(existing_engineered))

    if "time" in df.columns:
        buggy_start = pd.to_datetime(df["time"].min(), unit="s")
        buggy_end = pd.to_datetime(df["time"].max(), unit="s")
        print(f"Raw Unix interpretation: {buggy_start} -> {buggy_end}")

    # The source contains one reading per minute, but its time counter advances by 1.
    # Therefore this is explicitly a reconstructed timeline, not a trusted timestamp.
    df["datetime"] = pd.date_range("2016-01-01 00:00:00", periods=len(df), freq="1min")
    fixed_start, fixed_end = df["datetime"].min(), df["datetime"].max()
    print(f"Reconstructed one-minute timeline: {fixed_start} -> {fixed_end}")

    section("STEP 2: CLEANING")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Missing values: None")
    else:
        print("Missing values before imputation:\n", missing)
        numeric = df.select_dtypes(include=np.number).columns
        text = df.select_dtypes(exclude=np.number).columns
        for col in numeric:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in text:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    duplicate_rows = int(df.duplicated().sum())
    print("Duplicate rows:", duplicate_rows)
    if duplicate_rows:
        df = df.drop_duplicates().reset_index(drop=True)
        # Rebuild timeline after dropping duplicates so intervals remain regular.
        df["datetime"] = pd.date_range("2016-01-01", periods=len(df), freq="1min")
        fixed_start, fixed_end = df["datetime"].min(), df["datetime"].max()

    df = remove_duplicate_columns(df)
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print("Constant columns:", constant_cols if constant_cols else "None")
    if constant_cols:
        df = df.drop(columns=constant_cols)

    section("STEP 3: FEATURE ENGINEERING")
    if not ENERGY_COLUMNS.issubset(df.columns):
        raise KeyError(f"Dataset must contain {sorted(ENERGY_COLUMNS)}")

    # Only raw per-circuit columns are used. Grouped outputs are explicitly excluded.
    appliance_features = [
        c for c in df.columns
        if "[kw]" in c.lower()
        and c not in ENERGY_COLUMNS
        and c not in {"Kitchen [kW]", "Furnace [kW]"}
    ]
    weather_keywords = ["temperature", "humidity", "visibility", "pressure", "windspeed",
                        "cloud", "dew", "precip", "windbearing"]
    weather_features = [c for c in df.columns if any(k in c.lower() for k in weather_keywords)]

    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["dayofweek"] = df["datetime"].dt.dayofweek.astype("int8")
    df["month"] = df["datetime"].dt.month.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["season"] = to_season(df["month"])
    df["time_period"] = to_time_period(df["hour"])

    kitchen_cols = [c for c in appliance_features if "kitchen" in c.lower()]
    furnace_cols = [c for c in appliance_features if "furnace" in c.lower()]
    other_appliance_cols = [c for c in appliance_features if c not in kitchen_cols + furnace_cols]

    grouped_appliance_cols: list[str] = []
    if kitchen_cols:
        df["Kitchen [kW]"] = df[kitchen_cols].sum(axis=1)
        grouped_appliance_cols.append("Kitchen [kW]")
    if furnace_cols:
        df["Furnace [kW]"] = df[furnace_cols].sum(axis=1)
        grouped_appliance_cols.append("Furnace [kW]")
    grouped_appliance_cols.extend(other_appliance_cols)

    df["total_appliance"] = df[appliance_features].sum(axis=1)
    print("Raw appliance circuits:", appliance_features)
    print("Grouped appliance columns:", grouped_appliance_cols)

    section("STEP 4: ANOMALY LABEL FOR EDA")
    target = "use [kW]"
    mu, sigma = df[target].mean(), df[target].std()
    k_sigma = 3
    threshold = float(mu + k_sigma * sigma)
    df["_eda_anomaly"] = (df[target] > threshold).astype("int8")
    n_anomalies = int(df["_eda_anomaly"].sum())
    anomaly_rate = float(df["_eda_anomaly"].mean() * 100)
    q1, q3 = df[target].quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_outliers = int(((df[target] < q1 - 1.5 * iqr) | (df[target] > q3 + 1.5 * iqr)).sum())
    print(f"Threshold: {threshold:.4f} kW")
    print(f"Proxy anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
    print(f"IQR outliers: {iqr_outliers:,}")

    section("STEP 5: SAVE CLEANED DATASET")
    save_df = df.drop(columns="_eda_anomaly")
    tmp_zip = ZIP_PATH.with_suffix(".tmp.zip")
    save_df.to_csv(
        tmp_zip,
        index=False,
        compression={"method": "zip", "archive_name": "HomeC_cleaned_final.csv"},
    )
    os.replace(tmp_zip, ZIP_PATH)
    if CSV_PATH.exists() and source != CSV_PATH:
        CSV_PATH.unlink()
    print(f"Saved compressed dataset: {ZIP_PATH} ({save_df.shape[0]:,} rows, {save_df.shape[1]} cols)")

    section("STEP 6: APPLIANCE AND WEATHER TABLES")
    # One-minute kW readings -> kWh by multiplying by 1/60 hour.
    totals_kwh = (df[grouped_appliance_cols].sum() * SAMPLE_INTERVAL_HOURS).sort_values(ascending=False)
    share = (totals_kwh / totals_kwh.sum() * 100).round(2)
    peak_hour = {c: int(df.groupby("hour")[c].mean().idxmax()) for c in grouped_appliance_cols}
    peak_date = {
        c: str((df.set_index("datetime")[c].resample("D").sum() * SAMPLE_INTERVAL_HOURS).idxmax().date())
        for c in grouped_appliance_cols
    }
    top5_df = pd.DataFrame({
        "Total_kWh": totals_kwh,
        "Share_%": share,
        "Peak_Hour": pd.Series(peak_hour),
        "Peak_Date": pd.Series(peak_date),
    }).sort_values("Total_kWh", ascending=False)
    top5_df.to_csv(TOP5_PATH)
    print(top5_df.head(5).round(2))

    daily_weather = df.set_index("datetime").resample("D")[["use [kW]", "gen [kW]"] + weather_features].mean(numeric_only=True)
    corr_daily = daily_weather.corr()[["use [kW]", "gen [kW]"]].drop(["use [kW]", "gen [kW]"], errors="ignore")
    corr_daily.to_csv(CORR_PATH)

    section("STEP 7: VISUALIZATIONS")
    appliance_palette = sns.color_palette("Set2", max(len(totals_kwh), 1))
    appliance_color_map = {c: appliance_palette[i] for i, c in enumerate(totals_kwh.index)}
    top5 = top5_df.head(5)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(top5.index, top5["Total_kWh"], color=[appliance_color_map[c] for c in top5.index])
    for bar, pct in zip(bars, top5["Share_%"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{pct}%", ha="center", va="bottom", fontweight="bold")
    ax.set(title="Top 5 Energy-Consuming Appliances (2016)", xlabel="Appliance", ylabel="Energy (kWh)")
    ax.tick_params(axis="x", rotation=20)
    save_chart(fig, "chart6_top5_appliances_bar.png")

    others = totals_kwh.iloc[5:].sum()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(list(top5["Total_kWh"]) + [others], labels=list(top5.index) + ["Others"], autopct="%1.1f%%",
           colors=[appliance_color_map[c] for c in top5.index] + ["#B0B0B0"], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.set_title("Appliance Energy Share - Top 5 vs Others")
    save_chart(fig, "chart7_appliance_pie.png")

    # Simple single weather-vs-energy scatter (temperature vs use), replaces the
    # old 4-panel regression grid + separate correlation heatmap for an easier read.
    if "temperature" in daily_weather.columns:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sns.regplot(data=daily_weather, x="temperature", y="use [kW]", ax=ax,
                    scatter_kws={"alpha": 0.5, "s": 18, "color": COLOR_ENERGY},
                    line_kws={"color": COLOR_ANOMALY})
        r = daily_weather[["temperature", "use [kW]"]].corr().iloc[0, 1]
        ax.set(title=f"Temperature vs Energy Use (r={r:.2f})",
               xlabel="Temperature (°F)", ylabel="Power consumption (kW)")
        save_chart(fig, "chart5_weather_scatter.png")

    daily_energy = df.set_index("datetime")[["use [kW]", "gen [kW]"]].resample("D").sum() * SAMPLE_INTERVAL_HOURS
    hourly_avg = df.groupby("hour")[["use [kW]"]].mean()
    dow_avg = df.groupby("dayofweek")[["use [kW]"]].mean()
    month_avg = df.groupby("month")[["use [kW]"]].mean()

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(daily_energy.index, daily_energy["use [kW]"], color=COLOR_ENERGY, linewidth=1.3, label="Daily energy")
    yearly_daily_avg = daily_energy["use [kW]"].mean()
    ax.axhline(yearly_daily_avg, color=COLOR_ANOMALY, linestyle="--", linewidth=1,
               label=f"Average ({yearly_daily_avg:.1f} kWh/day)")
    ax.set(title="Household Energy Consumption Over 2016", xlabel="Date", ylabel="Energy (kWh/day)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.legend()
    save_chart(fig, "chart1_timeseries.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(df[target], bins=60, color=COLOR_ENERGY, edgecolor="white")
    ax.axvline(threshold, color=COLOR_ANOMALY, linestyle="--", label=f"Threshold={threshold:.2f} kW")
    ax.set(title="Distribution of One-Minute Power Consumption", xlabel="Power (kW)", ylabel="Readings")
    ax.legend(); save_chart(fig, "chart2_histogram.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.boxplot(x=df[target], ax=ax, color=COLOR_ENERGY)
    ax.axvline(threshold, color=COLOR_ANOMALY, linestyle="--", label=f"Threshold={threshold:.2f} kW")
    ax.set(title="Boxplot of Power Consumption", xlabel="Power (kW)"); ax.legend()
    save_chart(fig, "chart3_boxplot.png")

    low_t, high_t = hourly_avg[target].quantile([0.33, 0.66])
    colors = [COLOR_NORMAL if v <= low_t else COLOR_MEDIUM if v <= high_t else COLOR_HIGH for v in hourly_avg[target]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hourly_avg.index, hourly_avg[target], color=colors)
    ax.set(title="Average Power Consumption by Hour", xlabel="Hour", ylabel="Average power (kW)")
    ax.set_xticks(range(24)); save_chart(fig, "chart8_hourly_consumption.png")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.fill_between(daily_energy.index, daily_energy["use [kW]"], color=COLOR_ENERGY, alpha=0.4, label="Use")
    ax.plot(daily_energy.index, daily_energy["use [kW]"], color=COLOR_ENERGY, linewidth=1)
    ax.fill_between(daily_energy.index, daily_energy["gen [kW]"], color=COLOR_SOLAR, alpha=0.5, label="Solar")
    ax.plot(daily_energy.index, daily_energy["gen [kW]"], color=COLOR_SOLAR, linewidth=1)
    ax.set(title="Daily Energy Consumption vs Solar Generation", xlabel="Date", ylabel="Energy (kWh/day)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.legend()
    save_chart(fig, "chart9_daily_area.png")

    # Simple grouped bar of the top appliance's monthly total, replaces the old
    # multi-series stacked area chart (easier to read at a glance).
    monthly_appl_mwh = df.groupby("month")[grouped_appliance_cols].sum() * SAMPLE_INTERVAL_HOURS / 1000.0
    top_appliance = totals_kwh.index[0]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if top_appliance in monthly_appl_mwh.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([month_labels[m - 1] for m in monthly_appl_mwh.index],
               monthly_appl_mwh[top_appliance], color=COLOR_ENERGY)
        ax.set(title=f"Monthly Energy - {top_appliance} (Top Appliance)",
               xlabel="Month", ylabel="Energy (MWh)")
        save_chart(fig, "chart10_stacked_area_appliances.png")

    sample_df = df.iloc[::200]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(sample_df["datetime"], sample_df[target], color=COLOR_ENERGY, linewidth=0.8, label="Consumption")
    anomalies_sample = sample_df[sample_df["_eda_anomaly"] == 1]
    ax.scatter(anomalies_sample["datetime"], anomalies_sample[target], color=COLOR_ANOMALY, s=14, label="Anomaly")
    ax.axhline(threshold, color=COLOR_ANOMALY, linestyle="--", linewidth=1, label=f"Threshold={threshold:.2f} kW")
    ax.set(title=f"Proxy Anomaly Visualization - {n_anomalies:,} rows ({anomaly_rate:.2f}%)",
           xlabel="Date", ylabel="Power (kW)")
    ax.legend(); save_chart(fig, "chart12_anomaly_timeseries.png")

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(dow_labels[:len(dow_avg)], dow_avg[target], color=[COLOR_HIGH if d >= 5 else COLOR_ENERGY for d in dow_avg.index])
    ax.set(title="Average Power Consumption by Day of Week", xlabel="Day", ylabel="Average power (kW)")
    save_chart(fig, "chart13_dow_consumption.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([month_labels[m-1] for m in month_avg.index], month_avg[target], color=COLOR_ENERGY)
    ax.set(title="Average Power Consumption by Month", xlabel="Month", ylabel="Average power (kW)")
    save_chart(fig, "chart14_monthly_consumption.png")

    section("STEP 8 - KPI SUMMARY")
    kpi = {
        "sample_interval_minutes": SAMPLE_INTERVAL_MINUTES,
        "timeline_note": "Reconstructed at one-minute intervals because the raw time counter increments by 1.",
        "total_energy_kwh": round(float(df["use [kW]"].sum() * SAMPLE_INTERVAL_HOURS), 2),
        "average_power_kw": round(float(df["use [kW]"].mean()), 4),
        "average_energy_kw": round(float(df["use [kW]"].mean()), 4),  # backward-compatible key
        "peak_consumption_kw": round(float(df["use [kW]"].max()), 4),
        "peak_consumption_time": str(df.loc[df["use [kW]"].idxmax(), "datetime"]),
        "number_of_anomalies": n_anomalies,
        "anomaly_rate_pct": round(anomaly_rate, 2),
        "anomaly_threshold_kw": round(threshold, 4),
        "total_solar_generation_kwh": round(float(df["gen [kW]"].sum() * SAMPLE_INTERVAL_HOURS), 2),
        "date_range_start": str(fixed_start),
        "date_range_end": str(fixed_end),
        "top5_appliances": top5["Total_kWh"].round(2).to_dict(),
        "weather_correlation_use": corr_daily["use [kW]"].round(3).to_dict(),
        "weather_correlation_gen": corr_daily["gen [kW]"].round(3).to_dict(),
    }
    KPI_PATH.write_text(json.dumps(kpi, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(kpi, indent=2, ensure_ascii=False))
    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()
