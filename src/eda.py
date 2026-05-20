# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("logs", exist_ok=True)

def load_data(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nNull Values:\n{df.isnull().sum()}")
    return df

def clean_data(df):
    df = df.copy()
    # Fix TotalCharges stored as string
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"\nAfter cleaning — Shape: {df.shape}")
    return df

def plot_churn_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Count plot
    df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "crimson"])
    axes[0].set_title("Churn Count")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Pie chart
    df["Churn"].value_counts().plot(kind="pie", ax=axes[1],
                                     autopct="%1.1f%%", colors=["steelblue", "crimson"])
    axes[1].set_title("Churn Proportion")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("logs/churn_distribution.png", dpi=150)
    plt.close()
    print("Saved: logs/churn_distribution.png")

def plot_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("logs/correlation_matrix.png", dpi=150)
    plt.close()
    print("Saved: logs/correlation_matrix.png")

def plot_feature_vs_churn(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        df.groupby("Churn")[col].plot(kind="kde", ax=ax, legend=True)
        ax.set_title(f"{col} by Churn")
        ax.set_xlabel(col)

    plt.tight_layout()
    plt.savefig("logs/features_vs_churn.png", dpi=150)
    plt.close()
    print("Saved: logs/features_vs_churn.png")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    plot_churn_distribution(df)
    plot_correlation(df)
    plot_feature_vs_churn(df)
    print("\nEDA complete. Plots saved in logs/")
