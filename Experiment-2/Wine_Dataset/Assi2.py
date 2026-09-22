'''Plot Boxplots for all numerical attributes in the Wine dataset to visually identify any potential
outliers.'''
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df["target"]=wine.target

plt.figure(figsize=(7,5))
sns.boxplot(data=df)
plt.title("Boxplots of All Numerical Attributes")
plt.xlabel("Numerical Aattributes")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()