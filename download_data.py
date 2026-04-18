import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

column_names = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Primary URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

try:
    df = pd.read_csv(url, names=column_names, na_values="?")
    print("Downloaded from UCI successfully.")
except Exception as e:
    print(f"UCI failed: {e}")
    print("Trying backup source...")
    # Backup: same dataset hosted on GitHub
    backup_url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
    df = pd.read_csv(backup_url)
    df.columns = column_names[:len(df.columns)]
    print("Downloaded from backup successfully.")

print(f"Shape: {df.shape}")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")

df.to_csv("data/heart.csv", index=False)
print("\nSaved to data/heart.csv")