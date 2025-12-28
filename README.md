
# Fraud Detection for E-commerce and Banking

### **Project for Adey Innovations Inc.**

## ## Project Overview

This project focuses on building a robust fraud detection system for **Adey Innovations Inc.** to identify fraudulent activities in e-commerce and banking transactions. The primary goal is to address the extreme class imbalance typical of fraud data while balancing the trade-off between transaction security and user experience.

## ## Business Context

Adey Innovations Inc. aims to minimize financial losses due to fraud and build trust with customers. A key challenge is managing **False Positives** (alienating legitimate customers) and **False Negatives** (financial loss). Our evaluation focuses on metrics like **F1-Score** and **AUC-PR** to ensure the models effectively handle these competing costs.

## ## Key Features & Methodology

### **1. Data Analysis and Preprocessing**

* **Geolocation Integration:** Converted IP addresses to integers and used `merge_asof` for range-based lookups to map transactions to countries.
* **Feature Engineering:** * Calculated `time_since_signup` (duration between signup and purchase).
* Extracted `hour_of_day` and `day_of_week` to capture temporal fraud patterns.
* Developed `device_transaction_count` to track transaction velocity per device.


* **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to improve the model's ability to detect the minority fraud class.

### **2. Modeling and Evaluation**

We implemented a baseline Logistic Regression model and an advanced Random Forest Ensemble.

* **Baseline:** Logistic Regression.
* **Ensemble:** Random Forest Classifier with Stratified K-Fold Cross-Validation.

## ## Performance Results

Based on the latest model runs in `modeling.ipynb`:

| Model | F1-Score | AUC-PR | ROC-AUC |
| --- | --- | --- | --- |
| **Logistic Regression** | 0.5284 | 0.5120 | 0.7845 |
| **Random Forest** | **0.8583** | **0.7132** | **0.8446** |

**Note:** The Random Forest model significantly outperformed the baseline, achieving a mean F1-score of **0.8583** across 5 folds with a very low standard deviation (0.0019), indicating high model stability.

## ## Project Structure

```text
├── data/
│   ├── raw/                   # Fraud_Data.csv & IpAddress_to_Country.csv
│   └── processed/             # Cleaned datasets
├── notebooks/
│   ├── eda-fraud-data.ipynb   # Univariate/Bivariate Analysis
│   └── modeling.ipynb         # Model training and evaluation
├── src/
│   ├── preprocessing.py       # Data cleaning and feature engineering classes
│   ├── modeling.py            # Model training and evaluation classes
│   └── eda_utils.py           # Visualization utilities
├── requirements.txt           # Project dependencies
└── README.md

```

## ## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/muhajirhualis/KAIM_Week5-6

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Analysis:**
Navigate to the `notebooks/` folder and run `eda-fraud-data.ipynb` for the initial analysis or `modeling.ipynb` to train the models.

## ## Dependencies

* `pandas`, `numpy`
* `scikit-learn`
* `imblearn` (for SMOTE)
* `matplotlib`, `seaborn` (for visualization)

---

**Author:** Muhajer Hualis

**Date:** December 2025

**Submission:** Interim 2 - 10 Academy KAIM Week 5&6