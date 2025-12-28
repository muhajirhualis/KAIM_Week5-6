import unittest
import pandas as pd
import sys
import os

# Ensure the project root is in the path so 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import FraudPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        data = {
            'user_id': [1],
            'signup_time': ['2023-01-01 10:00:00'],
            'purchase_time': ['2023-01-01 12:00:00'],
            'device_id': ['D1'],
            'ip_address': ['123.45.67.89']
        }
        self.df = pd.DataFrame(data)
        self.processor = FraudPreprocessor(self.df)

    def test_clean_data(self):
        df_cleaned = self.processor.clean_data()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_cleaned['signup_time']))

if __name__ == '__main__':
    unittest.main()