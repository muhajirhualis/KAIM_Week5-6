import pandas as pd


class GeoFeatureEngineer:
    """
    Handles IP address processing and country mapping
    """

    def __init__(self, fraud_df: pd.DataFrame, ip_df: pd.DataFrame):
        self.fraud_df = fraud_df
        self.ip_df = ip_df

    def convert_ip_to_int(self):
        self.fraud_df["ip_address"] = self.fraud_df["ip_address"].astype(int)
        self.ip_df["lower_bound_ip_address"] = self.ip_df["lower_bound_ip_address"].astype(int)
        self.ip_df["upper_bound_ip_address"] = self.ip_df["upper_bound_ip_address"].astype(int)

    def merge_country(self) -> pd.DataFrame:
        self.ip_df = self.ip_df.sort_values("lower_bound_ip_address")
        self.fraud_df = self.fraud_df.sort_values("ip_address")

        merged_df = pd.merge_asof(
            self.fraud_df,
            self.ip_df,
            left_on="ip_address",
            right_on="lower_bound_ip_address",
            direction="backward"
        )
        return merged_df

