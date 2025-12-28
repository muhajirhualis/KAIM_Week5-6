
## Fraud Detection for E-commerce and Banking

### Project Overview

This project focuses on building a robust fraud detection system for **Adey Innovations Inc.** to identify fraudulent activities in e-commerce and banking transactions. We address extreme class imbalance while balancing the trade-off between security and user experience.

### Business Context

Adey Innovations Inc. aims to minimize financial losses due to fraud and build trust with customers. A key challenge is managing **False Positives** (alienating legitimate customers) and **False Negatives** (financial loss). Our evaluation focuses on **F1-Score** and **AUC-PR** to ensure the models effectively handle these competing costs.

### Key Features & Methodology

#### **1. Data Analysis and Preprocessing**

* **Geolocation Integration:** Converted IP addresses to integers and used `merge_asof` for range-based lookups to map transactions to countries.
* **Feature Engineering:** * Calculated `time_since_signup` (duration between signup and purchase).
* Extracted temporal patterns (`hour_of_day`, `day_of_week`).
* Developed `device_transaction_count` to track transaction velocity.


* **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to improve detection of the minority fraud class.

#### **2. Modeling & Hyperparameter Tuning**

We implemented a multi-stage modeling pipeline:

* **Baseline:** Logistic Regression for interpretability.
* **Ensemble:** Random Forest Classifier.
* **Tuning:** Used **GridSearchCV** to optimize hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`), ensuring the model generalizes well beyond default settings.
* **Validation:** Employed **Stratified K-Fold Cross-Validation** (5 folds) to confirm stability.

### Model Comparison & Selection

#### **Side-by-Side Performance**

The table below summarizes the performance of our models on the test set:

| Model | F1-Score | AUC-PR | ROC-AUC |
| --- | --- | --- | --- |
| **Logistic Regression** | 0.5284 | 0.5120 | 0.7845 |
| **Random Forest (Tuned)** | **0.8583** | **0.7132** | **0.8446** |

### **Model Selection Rationale**

We selected the **Tuned Random Forest** as the final model for the following reasons:

1. **Superior Detection Power:** It achieved a significant improvement in **F1-Score (0.858)** compared to the baseline, meaning it better balances precision and recall.
2. **Imbalance Handling:** The **AUC-PR of 0.713** demonstrates that the model remains effective even with the extreme scarcity of fraud cases.
3. **Stability:** Cross-validation resulted in a very low standard deviation (0.0019), indicating the model is robust and not overfitted to specific data segments.

### Project Structure

```text
├── data/
│   ├── raw/                   # Fraud_Data.csv & IpAddress_to_Country.csv
│   └── processed/             # Cleaned datasets
├── notebooks/
│   ├── eda-fraud-data.ipynb   # Univariate/Bivariate Analysis
│   └── modeling.ipynb         # Tuning, Comparison, and Selection
├── src/
│   ├── preprocessing.py       # Data cleaning & Feature Engineering classes
│   ├── modeling.py            # Modeler class with GridSearchCV & Metrics
│   └── eda_utils.py           # Visualization utilities
├── tests/
│   └── test_preprocessing.py  # Unit tests for the data pipeline
├── requirements.txt           # Project dependencies
└── README.md

```

### Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/muhajirhualis/KAIM_Week5-6

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Analysis:**
Run `notebooks/modeling.ipynb` to view the hyperparameter tuning process and final model comparison.

---

