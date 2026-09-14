import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df["target"]=wine.target

# Basic Data Exploration

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
#2.Plot Boxplots for all numerical attributes in the Wine dataset to visually identify any potential outliers.
plt.figure(figsize=(7,5))

sns.boxplot(data=df)

plt.title("Boxplots of All Numerical Attributes")
plt.xlabel("Numerical Aattributes")
plt.ylabel("Values")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#3. Heat Map
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot =True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap of wine Dataset")
plt.tight_layout()
plt.show()
