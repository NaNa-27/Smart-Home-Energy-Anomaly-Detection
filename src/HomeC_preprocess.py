# ==========================================================
# DAP391m PROJECT
# DAY 1 COMPLETE
# Member B
# Smart Home Energy Anomaly Detection
# ==========================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "HomeC_cleaned.csv"))
print(df.columns.tolist())
print("Dataset Loaded Successfully")
print()
# ==========================================================
# CREATE VISUALIZATION FOLDER
# ==========================================================

os.makedirs("visualization", exist_ok=True)

print("Visualization folder ready.")

print(df.columns.tolist())
print("Dataset Loaded Successfully")
print()

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)

print("\n")

print("=" * 60)
print("DATASET INFO")
print("=" * 60)

print(df.info())

print("\n")

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

# ==========================================================
# DATETIME CONVERSION
# ==========================================================

print("\nConverting Unix Timestamp...")

df["datetime"] = pd.to_datetime(df["time"], unit="s")

print(df[["time", "datetime"]].head())

# ==========================================================
# FEATURE IDENTIFICATION
# ==========================================================

print("\nFEATURE IDENTIFICATION")
print("=" * 60)

energy_features = [
    col for col in df.columns
    if "use" in col.lower()
    or "gen" in col.lower()
    or "overall" in col.lower()
]

weather_keywords = [
    "temperature",
    "humidity",
    "visibility",
    "pressure",
    "windspeed",
    "cloud",
    "dew",
    "precip"
]

weather_features = []

for col in df.columns:
    for keyword in weather_keywords:
        if keyword in col.lower():
            weather_features.append(col)

appliance_features = [
    col for col in df.columns
    if "[kW]" in col
    and col not in energy_features
]

print("\nEnergy Features:")
print(energy_features)

print("\nAppliance Features:")
print(appliance_features)

print("\nWeather Features:")
print(weather_features)
# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n")
print("=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("Missing values processed.")

# ==========================================================
# DATA CONSISTENCY CHECK
# ==========================================================

print("\n")
print("=" * 60)
print("DATA CONSISTENCY CHECK")
print("=" * 60)

negative_values = {}

for col in numeric_cols:

    count = (df[col] < 0).sum()

    if count > 0:
        negative_values[col] = count

if len(negative_values) == 0:
    print("No negative values detected.")
else:
    print("Columns containing negative values:")
    print(negative_values)

# ==========================================================
# DUPLICATE RECORDS
# ==========================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

# ==========================================================
# DUPLICATE COLUMNS
# ==========================================================

duplicate_columns = []

for i in range(len(df.columns)):
    for j in range(i + 1, len(df.columns)):
        if df.iloc[:, i].equals(df.iloc[:, j]):
            duplicate_columns.append(df.columns[j])

print("\nDuplicate Columns:")
print(duplicate_columns)

if len(duplicate_columns) > 0:
    df.drop(columns=duplicate_columns, inplace=True)

# ==========================================================
# CONSTANT COLUMNS
# ==========================================================

constant_cols = []

for col in df.columns:
    if df[col].nunique() <= 1:
        constant_cols.append(col)

print("\nConstant Columns:")
print(constant_cols)

if len(constant_cols) > 0:
    df.drop(columns=constant_cols, inplace=True)

# ==========================================================
# OUTLIER ANALYSIS
# ==========================================================

print("\n")
print("=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

target = "use [kW]"

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df[target] < lower) |
    (df[target] > upper)
]

print("Number of Outliers:", len(outliers))

print(
    "Percentage of Outliers:",
    round(len(outliers) / len(df) * 100, 2),
    "%"
)

print("""
Insight:
A significant number of observations fall
outside the normal operating range.

Business Interpretation:
These records may represent abnormal energy
consumption events and are potential anomaly
candidates for future modeling.

Justification:
Outlier analysis helps identify unusual
consumption behavior before applying anomaly
detection algorithms.
""")
# ==========================================================
# SAVE CLEANED DATA
# ==========================================================

df.to_csv("HomeC_cleaned_final.csv", index=False)

print("\nCleaned dataset saved.")

# ==========================================================
# EDA
# ==========================================================

print("\n")
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ==========================================================
# CHART 1
# TIME SERIES
# ==========================================================

sample_df = df.iloc[::500]

plt.figure(figsize=(14, 6))

plt.plot(
    sample_df["datetime"],
    sample_df["use [kW]"]
)

plt.title("Household Energy Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("Energy Consumption (kW)")

plt.tight_layout()

plt.savefig(
    "visualization/chart1_timeseries.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
print("""
============================================================
CHART 1 REPORT
============================================================

Chart Description:
This chart illustrates household energy
consumption over time.

Key Insight:
Energy consumption fluctuates considerably
throughout the observation period.
Several consumption spikes are visible.

Business Interpretation:
These spikes may indicate abnormal appliance
usage, simultaneous device operation, or
unusual household activities.

Justification:
Time-series visualization helps reveal
trends, seasonality, and abnormal patterns
which are essential for anomaly detection.
""")

# ==========================================================
# CHART 2
# HISTOGRAM
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["use [kW]"],
    bins=50
)

plt.title("Distribution of Energy Consumption")
plt.xlabel("Energy Consumption (kW)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "visualization/chart2_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
============================================================
CHART 2 REPORT
============================================================

Chart Description:
Histogram of household energy consumption.

Key Insight:
The distribution is positively skewed.
Most observations occur at lower consumption
levels.

Business Interpretation:
Rare high-energy events may correspond to
abnormal behavior.

Justification:
Distribution analysis supports anomaly
threshold selection.
""")
# ==========================================================
# CHART 3
# BOXPLOT
# ==========================================================

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=df["use [kW]"]
)

plt.title("Boxplot of Energy Consumption")

plt.tight_layout()

plt.savefig(
    "visualization/chart3_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
============================================================
CHART 3 REPORT
============================================================

Chart Description:
Boxplot showing the spread of energy usage.

Key Insight:
Several extreme observations exist beyond
the whiskers.

Business Interpretation:
These observations may represent abnormal
consumption events.

Justification:
Boxplots are effective for identifying
potential anomalies.
""")
# ==========================================================
# CHART 4
# CORRELATION HEATMAP
# ==========================================================

corr_columns = [
    "use [kW]",
    "gen [kW]",
    "Dishwasher [kW]",
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Fridge [kW]",
    "Microwave [kW]",
    "temperature",
    "humidity",
    "pressure",
    "windSpeed"
]

corr_columns = [
    col for col in corr_columns
    if col in df.columns
]

sample_corr = df[corr_columns].sample(
    min(100000, len(df)),
    random_state=42
)

corr_matrix = sample_corr.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "visualization/chart4_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
============================================================
CHART 4 REPORT
============================================================

Chart Description:
Heatmap illustrating correlations among
energy, appliance, and weather variables.

Key Insight:
Several appliance variables show moderate
to strong correlations with total energy
consumption.

Business Interpretation:
Strongly correlated variables can serve as
important predictors in anomaly detection
models.

Justification:
Correlation analysis helps identify key
drivers of abnormal consumption.
""")

# ==========================================================
# SKEWNESS
# ==========================================================

print("\n")
print("=" * 60)
print("SKEWNESS ANALYSIS")
print("=" * 60)

print(
    "Skewness of use [kW]:",
    round(df["use [kW]"].skew(), 4)
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("DAY 1 COMPLETED")
print("=" * 60)

print("""
Completed Tasks:

✓ Dataset Overview
✓ Datetime Conversion
✓ Feature Identification
✓ Missing Value Detection
✓ Missing Value Treatment
✓ Data Consistency Check
✓ Duplicate Row Detection
✓ Duplicate Column Detection
✓ Garbage Column Detection
✓ Outlier Analysis
✓ Time Series Analysis
✓ Distribution Analysis
✓ Boxplot Analysis
✓ Correlation Analysis
✓ Cleaned Dataset Export

Dataset Ready For:

✓ Day 2 - EDA Refinement
✓ Day 3 - Anomaly Detection Modeling
✓ Final Report Writing
""")