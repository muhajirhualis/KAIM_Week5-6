from imblearn.over_sampling import SMOTE


class ImbalanceHandler:
    """
    Handles class imbalance using SMOTE
    """
    
    def apply_smote(self, X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
