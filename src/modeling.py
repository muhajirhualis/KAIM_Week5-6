import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE

class FraudModeler:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}

    def split_data(self, test_size=0.2):
        """
        Task 2a.1: Stratified train-test split.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")

    def handle_imbalance(self):
        """
        Task 1.6: Apply SMOTE to training data only.
        """
        print("Class distribution before SMOTE:", np.bincount(self.y_train))
        smote = SMOTE(random_state=42) # [cite: 135]
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print("Class distribution after SMOTE:", np.bincount(self.y_train))

    def train_baseline(self):
        """
        Task 2a.3: Train Logistic Regression Baseline.
        """
        print("Training Logistic Regression Baseline...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['LogisticRegression'] = lr
        return lr

    def train_random_forest(self):
        """
        Task 2b.1: Train Random Forest (Ensemble).
        """
        print("Training Random Forest...")
        # Basic hyperparameter tuning [cite: 151]
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = rf
        return rf

    def evaluate_model(self, model_name):
        """
        Task 2a.4: Evaluate using AUC-PR, F1, etc.
        """
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        # AUC-PR Calculation [cite: 58]
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        auc_pr = auc(recall, precision)
        
        print(f"--- {model_name} Evaluation ---")
        print(classification_report(self.y_test, y_pred))
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_prob):.4f}")
        print("-" * 30)
        
        return auc_pr

    def cross_validate(self, model_name, k=5):
        """
        Task 2b.4: Stratified K-Fold Cross Validation.
        """
        model = self.models[model_name]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) # [cite: 154]
        
        # Using f1 as scoring metric
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='f1')
        
        print(f"--- Cross Validation ({k}-Fold) for {model_name} ---")
        print(f"Mean F1 Score: {scores.mean():.4f}")
        print(f"Standard Deviation: {scores.std():.4f}") # [cite: 155]