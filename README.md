# Fraud Detection for E-commerce & Banking  
**Client**: Adey Innovations Inc  
**Goal**: Improve fraud detection accuracy while minimizing false positives.

## Business Need
Accurate fraud detection in e-commerce and credit card transactions to reduce financial loss and improve trust.

## Objective
Prepare clean, feature-rich datasets for fraud detection modeling.

## Tasks Completed
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Geolocation integration using IP mapping
- Feature engineering (time & velocity features)
- Data scaling and encoding
- Class imbalance handling using SMOTE

## Structure
- `src/` – modular OOP-based code
- `notebooks/` – EDA and feature engineering
- `data/` – raw and processed datasets (gitignored)


## Setup
```bash
git clone <your-repo-url>
cd fraud-detection
pip install -r requirements.txt
# Place datasets in data/raw/