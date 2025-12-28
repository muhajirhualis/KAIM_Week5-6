import pandas as pd
import numpy as np
import socket
import struct
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FraudPreprocessor:
    def __init__(self, fraud_df, ip_df=None):
        """
        Initializes the preprocessor with fraud data and optional IP mapping data.
        """
        self.df = fraud_df.copy()
        self.ip_df = ip_df.copy() if ip_df is not None else None
        self.preprocessor = None  # To store the sklearn pipeline

    def clean_data(self):
        """
        Task 1.1: Handle missing values, duplicates, and data types.
        """
        # Remove duplicates [cite: 115]
        self.df.drop_duplicates(inplace=True)
        
        # Correct data types [cite: 116]
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        
        # Handle missing values (e.g., drop only if critical info is missing) [cite: 114]
        # Justification: User ID and device ID are critical for tracking.
        self.df.dropna(subset=['user_id', 'device_id'], inplace=True)
        
        return self.df

    @staticmethod
    def _ip_to_int(ip):
        """Helper to convert IP string to integer[cite: 122]."""
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except (socket.error, TypeError):
            return 0

    def merge_geolocation(self):
            """
            Task 1.3: Merge Fraud_Data with IpAddress_to_Country using range-based lookup.
            """
            if self.ip_df is None:
                raise ValueError("IP Database dataframe not provided.")

            # 1. Convert IPs in fraud data to integers
            print("Converting IPs to integers...")
            self.df['ip_int'] = self.df['ip_address'].apply(self._ip_to_int)
            
            # 2. FIX: Explicitly cast IP ranges in the lookup table to integers
            # Pandas often reads large numbers as floats; we force them to int64 for the merge
            self.ip_df['lower_bound_ip_address'] = self.ip_df['lower_bound_ip_address'].astype('int64')
            self.ip_df['upper_bound_ip_address'] = self.ip_df['upper_bound_ip_address'].astype('int64')

            # 3. Sort both dataframes (Requirement for merge_asof)
            self.ip_df = self.ip_df.sort_values('lower_bound_ip_address')
            self.df = self.df.sort_values('ip_int')

            # 4. Merge
            print("Merging geolocation data...")
            self.df = pd.merge_asof(
                self.df,
                self.ip_df,
                left_on='ip_int',
                right_on='lower_bound_ip_address',
                direction='backward'
            )

            # 5. Filter out matches where IP is actually above the upper bound
            # (merge_asof only checks the lower bound, so we must verify the upper bound)
            mask = self.df['ip_int'] > self.df['upper_bound_ip_address']
            self.df.loc[mask, 'country'] = 'Unknown'
            
            # Fill remaining NaNs
            self.df['country'] = self.df['country'].fillna('Unknown')
            
            return self.df

    def feature_engineering(self):
        """
        Task 1.4: Create time-based and frequency features.
        """
        # Time-based features [cite: 127-129]
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        
        # Time since signup (in seconds) [cite: 130]
        self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds()
        
        # Transaction frequency (velocity) per device [cite: 126]
        # Calculate how many transactions per device occur in the dataset
        device_freq = self.df.groupby('device_id')['user_id'].transform('count')
        self.df['device_transaction_count'] = device_freq

        return self.df

    def transform_data(self, fit=True):
        """
        Task 1.5: Encode and scale features.
        """
        # Define columns
        categorical_features = ['source', 'browser', 'sex', 'country']
        numerical_features = ['purchase_value', 'time_since_signup', 'age', 'device_transaction_count']
        
        # Pipeline setup
        if self.preprocessor is None:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features), # [cite: 132]
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # [cite: 133]
                ]
            )
        
        # Fit-transform or just transform
        if fit:
            X_processed = self.preprocessor.fit_transform(self.df)
        else:
            X_processed = self.preprocessor.transform(self.df)
            
        # Get feature names for clarity later
        # Note: OneHotEncoder feature names can be complex to retrieve in older sklearn versions
        try:
             cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
             feature_names = numerical_features + list(cat_names)
        except AttributeError:
             feature_names = None # Fallback

        return X_processed, self.df['class'], feature_names