import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FraudEDA:
    def __init__(self, df):
        """
        Initializes the EDA class with the dataframe.
        """
        self.df = df
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_class_distribution(self, target_col='class'):
        """
        Task 1.2: Quantify and visualize class imbalance[cite: 120].
        """
        count = self.df[target_col].value_counts()
        percentage = self.df[target_col].value_counts(normalize=True) * 100
        
        print(f"Class Distribution:\n{count}")
        print(f"\nPercentage:\n{percentage}")
        
        plt.figure(figsize=(6, 4))
        # Log scale is often useful for highly imbalanced data
        sns.countplot(x=target_col, data=self.df, palette='viridis')
        plt.title(f'Class Distribution (Target: {target_col})')
        plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
        plt.ylabel('Count (Log Scale)')
        plt.yscale('log') 
        plt.show()

    def plot_numerical_distributions(self, features):
        """
        Task 1.2: Univariate analysis for numerical features[cite: 118].
        """
        for col in features:
            if col not in self.df.columns:
                continue
                
            plt.figure(figsize=(12, 5))
            
            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
            
            # Boxplot to visualize outliers
            plt.subplot(1, 2, 2)
            sns.boxplot(y=self.df[col], color='lightgreen')
            plt.title(f'Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()

    def plot_categorical_distributions(self, features):
        """
        Task 1.2: Univariate analysis for categorical features[cite: 118].
        """
        for col in features:
            if col not in self.df.columns:
                continue
            
            # Limit to top 10 categories for readability if high cardinality
            top_cats = self.df[col].value_counts().nlargest(10).index
            filtered_data = self.df[self.df[col].isin(top_cats)]
            
            plt.figure(figsize=(10, 5))
            sns.countplot(y=col, data=filtered_data, order=top_cats, palette='magma')
            plt.title(f'Top 10 Categories in {col}')
            plt.show()

    def plot_bivariate_analysis(self, target_col, features):
        """
        Task 1.2: Relationships between features and target[cite: 119].
        """
        for col in features:
            if col not in self.df.columns:
                continue
            
            plt.figure(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Boxplot for numeric vs target
                sns.boxplot(x=target_col, y=col, data=self.df, palette='coolwarm')
                plt.title(f'{col} distribution by Class')
            else:
                # Stacked bar for categorical vs target (Normalized to show rates)
                # We calculate proportions to see if fraud is more prevalent in certain categories
                props = self.df.groupby(col)[target_col].value_counts(normalize=True).unstack()
                props.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(10, 6))
                plt.title(f'Proportion of Fraud by {col}')
                plt.ylabel('Proportion')
                plt.legend(title='Fraud Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()

    def correlation_heatmap(self, numerical_features):
        """
        Visualize correlations between numerical features to check for multicollinearity.
        """
        plt.figure(figsize=(12, 10))
        # Select only numeric columns present in the input list
        valid_features = [f for f in numerical_features if f in self.df.columns]
        corr = self.df[valid_features].corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()