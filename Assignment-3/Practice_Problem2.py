import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 27, 32, 38],
    "Salary": [30000, 40000, 50000, 60000, 75000, 90000, 120000, 45000, 65000, 85000],
    "Department": [
        "IT", "HR", "Finance", "IT", "Marketing",
        "Finance", "IT", "HR", "Marketing", "IT"
    ],
    "Years of Experience": [1, 2, 4, 5, 8, 12, 15, 3, 6, 10]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

df.loc[2, "Age"] = np.nan
df.loc[5, "Salary"] = np.nan
df.loc[7, "Department"] = np.nan
df.loc[8, "Years of Experience"] = np.nan

print("\n--- Dataset With Missing Values ---")
print(df)


print("\n--- Missing Values ---")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Years of Experience"] = df["Years of Experience"].fillna(
    df["Years of Experience"].median()
)
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

print("\n--- After Handling Missing Values ---")
print(df)

label_encoder = LabelEncoder()
df["Department"] = label_encoder.fit_transform(df["Department"])

scaler = MinMaxScaler()

numerical_columns = [
    "Age",
    "Salary",
    "Years of Experience"
]

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\n--- Final Preprocessed Dataset Using MinMaxScaler ---")
print(df)