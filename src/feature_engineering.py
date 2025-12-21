import pandas as pd


class FeatureEngineer:
    """
    Creates new meaningful features for fraud detection
    """
    def __init__(self, df):
        self.df = df

    def add_time_features(self):
        self.df["hour_of_day"] = self.df["purchase_time"].dt.hour
        self.df["day_of_week"] = self.df["purchase_time"].dt.dayofweek
        self.df["time_since_signup"] = (
            self.df["purchase_time"] - self.df["signup_time"]
        ).dt.total_seconds()
        return self.df

    def transaction_velocity(self):
        self.df["transaction_count"] = (
            self.df.groupby("user_id")["purchase_time"]
            .transform("count")
        )
        return self.df
