import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

class FraudExplainer:
    def __init__(self, model, X_test, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.feature_names = feature_names
        # TreeExplainer is fast to initialize
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = None 

    def plot_built_in_importance(self, top_n=10):
        """Task 3.1: Instant visualization using model internal importance."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Built-in Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()

    def plot_shap_summary(self, n_samples=200):
        """Task 3.2: Global feature importance using a robust subset."""
        # Use a subset for speed
        X_subset = self.X_test[:n_samples]
        raw_shap_values = self.explainer.shap_values(X_subset, check_additivity=False)
        
        # Robust selection of the 'Fraud' class (Class 1)
        if isinstance(raw_shap_values, list):
            # Standard list format [class_0_vals, class_1_vals]
            shap_to_plot = raw_shap_values[1]
        elif len(raw_shap_values.shape) == 3:
            # 3D array format (samples, features, classes)
            shap_to_plot = raw_shap_values[:, :, 1]
        else:
            # Fallback for single-output or collapsed arrays
            shap_to_plot = raw_shap_values

        shap.summary_plot(shap_to_plot, X_subset, feature_names=self.feature_names)

    def plot_individual_force(self, row_index, label_name):
        """Task 3.2: Calculate SHAP for a single specific row safely."""
        # Ensure input is 2D (1, num_features)
        specific_row = self.X_test[row_index].reshape(1, -1)
    
        # Calculate SHAP for just this one row
        raw_shap_values = self.explainer.shap_values(specific_row, check_additivity=False)
    
        # 1. Handle Expected Value (Base Value)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1:
            base_val = self.explainer.expected_value[1]
        else:
            base_val = self.explainer.expected_value

        # 2. Handle SHAP Values for Class 1 (Fraud)
        if isinstance(raw_shap_values, list):
            # It's a list: raw_shap_values[class_index][row_index]
            shap_vals = raw_shap_values[1][0]
        elif len(raw_shap_values.shape) == 3:
            # 3D Array: (samples, features, classes) -> (0, all_features, 1)
            shap_vals = raw_shap_values[0, :, 1]
        elif len(raw_shap_values.shape) == 2:
            # 2D Array: (samples, features)
            shap_vals = raw_shap_values[0]
        else:
            shap_vals = raw_shap_values

        # 3. Generate the plot
        # Force plots require specific_row to be 1D or matched to shap_vals
        shap.force_plot(
            base_val, 
            shap_vals, 
            specific_row[0], # Pass as 1D array for visualization
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot: {label_name}")
        plt.show()