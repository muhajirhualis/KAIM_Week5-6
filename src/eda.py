import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EDA:
    """
    Performs Exploratory Data Analysis:
    - Univariate analysis
    - Bivariate analysis
    - Class imbalance visualization
    """

    @staticmethod
    def univariate_analysis(df, column):
        plt.figure(figsize=(6,4))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.show()

    @staticmethod
    def bivariate_analysis(df, feature, target):
        plt.figure(figsize=(6,4))
        sns.boxplot(x=target, y=feature, data=df)
        plt.title(f"{feature} vs {target}")
        plt.show()

    @staticmethod
    def class_distribution(df, target):
        plt.figure(figsize=(5,4))
        df[target].value_counts().plot(kind="bar")
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.show()
