import pandas as pd
from pathlib import Path
import sys

# Ensure project root (the directory that contains "src") is on sys.path
_current_file = Path(__file__).resolve()
for _p in _current_file.parents:
    if (_p / "src").is_dir():
        sys.path.insert(0, str(_p))
        break
else:
    # Fallback: add parent of the scripts directory
    sys.path.insert(0, str(_current_file.parents[1]))


from src.data_cleaning import DataCleaner
from src.geo_features import GeoFeatureEngineer
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor


def main():
    print("Starting Task 1 Pipeline...")

    # Load data
    fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")

    # Cleaning
    cleaner = DataCleaner(fraud_df)
    fraud_df = cleaner.remove_duplicates()
    fraud_df = cleaner.handle_missing_values()
    fraud_df = cleaner.fix_data_types(
        ["signup_time", "purchase_time"]
    )

    # Geolocation
    geo = GeoFeatureEngineer(fraud_df, ip_df)
    geo.convert_ip_to_int()
    fraud_df = geo.merge_country()

    # Feature Engineering
    fe = FeatureEngineer(fraud_df)
    fraud_df = fe.add_time_features()
    fraud_df = fe.transaction_velocity()

    # Preprocessing
    pre = Preprocessor()
    categorical_cols = ["browser", "source", "sex", "country"]

    fraud_df = pre.encode_categorical_features(
        fraud_df,
        categorical_columns=categorical_cols
    )
    numeric_cols = [
        "age",
        "time_since_signup",
        "transaction_count"
    ]
    fraud_df = pre.scale_numeric_features(
        fraud_df,
        numeric_columns=numeric_cols
    )
    

    # Save processed data
    fraud_df.to_csv(
        "data/processed/fraud_processed.csv",
        index=False
    )

    print("Task 1 completed successfully!")

if __name__ == "__main__":
    main()
