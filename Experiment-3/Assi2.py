#Modify the pipeline code to apply MinMaxScaler instead of StandardScaler.
import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data={
    "Age":[21,20,23,None,25,31],
    "Salary":[35000,40000,57000,56000,None,60000],
    "Department":["IT","HR","Finance","Marketing","Sales","R & D"],
    "Year_of_Experience":[0,2,3,2,None,5]
}

df=pd.DataFrame(data)
print(df)
print(df.isnull().sum())

numeric_features=["Age","Salary","Year_of_Experience"]
categorical_features=["Department"]

numeric_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",MinMaxScaler())
])

categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor=ColumnTransformer(
    transformers=[
        ("num",numeric_transformer,numeric_features),
        ("cat",categorical_transformer,categorical_features)
    ]
)

x_processed=preprocessor.fit_transform(df)

print(x_processed)
print(x_processed.shape)