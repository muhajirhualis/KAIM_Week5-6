import unittest
import pandas as pd
from src.preprocessing import FraudPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a small dummy dataframe for testing
        data = {
            'user_id': [1, 2],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
            'purchase_time': ['2023-01-01 12:00:00', '2023-01-01 13:00:00'],
            'device_id': ['D1', 'D2'],
            'ip_address': ['123.45.67.89', '98.76.54.32']
        }
        self.df = pd.DataFrame(data)
        self.processor = FraudPreprocessor(self.df)

    def test_clean_data(self):
        df_cleaned = self.processor.clean_data()
        self.assertEqual(pd.api.types.is_datetime64_any_dtype(df_cleaned['signup_time']), True)
        self.assertEqual(df_cleaned.shape[0], 2)

if __name__ == '__main__':
    unittest.main()