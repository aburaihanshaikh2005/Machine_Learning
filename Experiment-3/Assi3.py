'''Compare and document the effect of StandardScaler vs MinMaxScaler by printing the transformed arrays and
observing the numerical ranges.'''
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.impute import SimpleImputer

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

imputer=SimpleImputer(strategy="median")
numeric_data=imputer.fit_transform(df[numeric_features])

#StandardScaler
standard_scaler=StandardScaler()
standard_data=standard_scaler.fit_transform(numeric_data)
print("\nStandardScaler Output:")
print(standard_data)
#Ranges
print("Minimum:",standard_data.min())
print("Maximum:",standard_data.max())

#MinMaxScaler
minmax_scaler=MinMaxScaler()
minmax_data=minmax_scaler.fit_transform(numeric_data)
print("\nMinMaxScaler Output:")
print(minmax_data)
#Ranges
print("Minimum:",minmax_data.min())
print("Maximum:",minmax_data.max())

 