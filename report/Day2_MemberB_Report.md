# Author: Bùi Đoàn Quang Hưng - SE19




# **DAP391m Final Project**

## **Day 2 – Member B**

### **Project Title**

**Anomaly Detection on Smart Home Energy Data**

---

# **1\. Problem Understanding**

## **1.1 Background**

The rapid growth of smart home technologies has enabled continuous monitoring of household energy consumption through interconnected sensors and smart devices. These systems generate large volumes of data that can be analyzed to improve energy efficiency, reduce operational costs, and enhance home automation capabilities.

However, abnormal energy consumption events may indicate equipment malfunction, inefficient energy usage, unusual occupant behavior, or sensor-related issues. Detecting such anomalies is essential because undetected abnormal patterns can lead to increased energy costs, reduced equipment lifespan, and inefficient resource utilization.

## **1.2 Motivation**

Traditional energy monitoring approaches often rely on manual inspection and predefined thresholds, which may fail to identify complex abnormal patterns in large-scale datasets. Machine learning and anomaly detection techniques provide a more effective solution by automatically discovering unusual consumption behaviors from historical data.

The availability of detailed appliance-level energy measurements and weather information in the HomeC dataset provides an opportunity to investigate the factors associated with abnormal electricity usage and evaluate the effectiveness of anomaly detection models.

## **1.3 Problem Definition**

This project aims to identify abnormal electricity consumption periods within a smart home environment using data analytics and machine learning techniques. The study focuses on analyzing energy consumption behavior, understanding relationships among appliance and weather variables, and supporting the development of anomaly detection models.

## **1.4 Research Objectives**

The main objectives of this project are:

* Identify factors associated with abnormal energy consumption.  
* Analyze energy consumption behavior through exploratory data analysis.  
* Support the development and evaluation of anomaly detection models.  
* Investigate the influence of weather conditions on household energy consumption.  
* Provide insights that can improve smart home energy management systems.

---

# **2\. Research Questions**

## **RQ1**

### **What factors are associated with abnormal energy consumption?**

This question investigates which appliance-related and weather-related variables contribute to unusual electricity usage patterns. Understanding these factors may help explain the occurrence of anomalies and improve feature selection for anomaly detection models.

---

## **RQ2**

### **Which anomaly detection model performs best on this dataset?**

This question compares the performance of multiple anomaly detection models using evaluation metrics such as F1-score and AUC-ROC. The objective is to identify the most effective model for detecting abnormal energy consumption events.

---

## **RQ3**

### **Does weather influence abnormal household energy consumption?**

This question examines the relationship between weather variables and household electricity usage. The analysis focuses on identifying whether changes in temperature, humidity, wind speed, or other environmental factors contribute to abnormal energy consumption behavior.

---

# **3\. Evaluation Metrics**

## **F1-Score**

F1-score is selected because anomaly detection datasets are typically highly imbalanced. In this project, abnormal events represent only a small portion of the observations. As a result, accuracy alone may provide misleading performance estimates.

F1-score combines Precision and Recall into a single metric and provides a balanced evaluation of classification performance. It is particularly useful when both false positives and false negatives are important considerations.

### **Formula**

F1 \= 2 × (Precision × Recall) / (Precision \+ Recall)

---

## **AUC-ROC**

The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) measures a model's ability to distinguish between normal and abnormal observations across different decision thresholds.

AUC-ROC is selected because it evaluates ranking performance rather than relying on a single threshold. This characteristic makes it particularly suitable for anomaly detection tasks where the optimal decision threshold may vary.

---

## **Why Accuracy Is Not Sufficient**

Accuracy may appear high even when a model completely fails to identify anomalies due to class imbalance. For example, if anomalies represent only 1% of observations, a model that predicts all observations as normal would still achieve approximately 99% accuracy.

Therefore, F1-score and AUC-ROC provide more reliable and meaningful performance evaluation.

---

# **4\. Data Understanding**

## **Dataset Overview**

The dataset used in this project is the Smart Home Dataset with Weather Information (HomeC) obtained from Kaggle.

### **Dataset Characteristics**

| Attribute | Value |
| ----- | ----- |
| Number of Rows | 503,910 |
| Number of Columns | 32 |
| Time Range | January 2016 – December 2016 |
| Data Type | Time-Series |
| Domain | Smart Home Energy Consumption |

---

## **Feature Categories**

### **Energy Consumption Features**

* use \[kW\]  
* gen \[kW\]  
* House overall \[kW\]

### **Appliance Features**

* Dishwasher \[kW\]  
* Furnace 1 \[kW\]  
* Furnace 2 \[kW\]  
* Home office \[kW\]  
* Fridge \[kW\]  
* Microwave \[kW\]  
* Living room \[kW\]  
* Garage door \[kW\]  
* Barn \[kW\]  
* Well \[kW\]

### **Weather Features**

* temperature  
* humidity  
* visibility  
* pressure  
* windSpeed  
* cloudCover  
* dewPoint  
* precipProbability  
* precipIntensity

---

# **5\. Data Cleaning**

## **Missing Value Analysis**

The dataset was inspected for missing values using `df.isnull().sum()`. No missing values were detected in any feature. Therefore, no imputation procedures were required.

### **Justification**

Missing value analysis ensures data completeness and prevents potential biases during model training.

---

## **Duplicate Record Detection**

Duplicate row analysis was conducted using `df.duplicated().sum()`.

### **Result**

* Duplicate Rows: 0

No duplicated observations were detected.

---

## **Duplicate Column Detection**

Several columns contained redundant information.

### **Identified Duplicate Columns**

* House overall \[kW\]  
* Solar \[kW\]

These columns were removed to reduce redundancy and prevent multicollinearity.

### **Justification**

Removing duplicated features improves computational efficiency and reduces unnecessary information.

---

## **Outlier Analysis**

The Interquartile Range (IQR) method was applied to identify extreme observations.

### **Result**

* Number of Outliers: 34,211

These observations were retained because they may represent meaningful abnormal energy consumption events.

### **Justification**

Outliers are particularly important in anomaly detection projects because they may correspond to genuine abnormal behaviors rather than measurement errors.

---

# **6\. Exploratory Data Analysis**

## **Figure 1\. Household Energy Consumption Over Time**

### **Chart Description**

This figure illustrates household electricity consumption over time.

### **Key Insight**

Energy consumption fluctuates considerably throughout the observation period. Several significant spikes are observed, indicating periods of unusually high electricity usage.

### **Business Interpretation**

These spikes may correspond to abnormal appliance usage or unusual household activities and therefore represent potential anomaly candidates.

### **Justification**

Time-series visualization helps identify trends, seasonality, and unusual consumption behavior.

---

## **Figure 2\. Distribution of Energy Consumption**

### **Chart Description**

Histogram showing the distribution of household energy consumption.

### **Key Insight**

The distribution is strongly right-skewed. Most observations occur at low consumption levels, while high-consumption events occur relatively infrequently.

### **Business Interpretation**

The rarity of high-consumption events suggests that anomaly detection methods are appropriate for this dataset.

### **Justification**

Distribution analysis provides insight into the statistical characteristics of energy consumption.

---

## **Figure 3\. Boxplot of Energy Consumption**

### **Chart Description**

Boxplot illustrating the spread of energy consumption values.

### **Key Insight**

Numerous observations exist beyond the whiskers, indicating the presence of extreme values and potential outliers.

### **Business Interpretation**

These observations may represent abnormal energy usage patterns and should be investigated further.

### **Justification**

Boxplots provide an efficient method for detecting outliers.

---

## **Figure 4\. Correlation Heatmap**

### **Chart Description**

Heatmap showing correlations among energy, appliance, and weather variables.

### **Key Insight**

Several appliance variables exhibit moderate to strong correlations with overall energy consumption. Weather variables demonstrate weaker but noticeable relationships.

### **Business Interpretation**

Highly correlated variables may serve as important predictors in anomaly detection models.

### **Justification**

Correlation analysis supports feature selection and model development.

---

# **7\. Summary of Findings**

The exploratory analysis revealed substantial variability in household energy consumption. Several pronounced consumption spikes and a large number of outliers were identified. The distribution of energy usage is strongly right-skewed, indicating that abnormal events occur relatively infrequently. Correlation analysis suggests that appliance-related variables contribute more significantly to energy consumption behavior than weather variables.

These findings support the use of anomaly detection techniques and provide valuable guidance for subsequent feature engineering and model development stages.

