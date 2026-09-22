'''Generate a correlation heatmap(using Seaborn's heatmap) and identify the pair of features
that exhibit the strongest positive correlation'''
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df["target"]=wine.target

plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot =True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap of wine Dataset")
plt.tight_layout()
plt.show()