import pandas as pd

class DataCleaner:
    """
    Handles basic data cleaning tasks:
    - Missing values
    - Duplicates
    - Data type corrections
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_duplicates(self) -> pd.DataFrame:
        self.df = self.df.drop_duplicates()
        return self.df

    def handle_missing_values(self) -> pd.DataFrame:
        # Drop rows with missing values (justified due to small proportion)
        self.df = self.df.dropna()
        return self.df

    def fix_data_types(self, datetime_cols=None) -> pd.DataFrame:
        if datetime_cols:
            for col in datetime_cols:
                self.df[col] = pd.to_datetime(self.df[col])
        return self.df
