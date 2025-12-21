import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Handles scaling and encoding
    """

    def scale_numeric_features(self, df, numeric_columns):
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df

    def encode_categorical_features(self, df, categorical_columns):
        """
        Encode only low-cardinality categorical features
        """
        df_encoded = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )
        return df_encoded

