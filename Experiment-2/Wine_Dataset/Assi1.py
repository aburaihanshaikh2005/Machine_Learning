#Load and perform a complete EDA on the Wine dataset available in Scikit-learn
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df["target"]=wine.target

print("---First Five Rows---")
print(df.head())

print("\n---Dataset Information")
print(df.info())

print("\n---Statistical Summary---")
print(df.describe())

print("\n---Missing values---")
print(df.isnull().sum())

print("\n---Correlation Matrix---")
print(df.corr(numeric_only=True))