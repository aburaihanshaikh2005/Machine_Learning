import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = np.array([
    [22, 30000],
    [25, 40000],
    [28, 50000],
    [30, 60000],
    [35, 75000]
])

print("--- Original Data ---")
print(data)

standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(data)

print("\n--- StandardScaler Output ---")
print(standard_data)

print("\nStandardScaler Range:")
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())


minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(data)

print("\n--- MinMaxScaler Output ---")
print(minmax_data)

print("\nMinMaxScaler Range:")
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())