'''1. Download a real-estate dataset and predict house prices based solely on the
 house area (in sq. ft).'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("house_prices_dataset.csv")

# Display first 5 rows
print(data.head())

# Select house area as input and price as output
X = data[['square_feet']]
y = data['price']

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


area = pd.DataFrame([[2000]], columns=['square_feet'])
predicted_price = model.predict(area)

print("Predicted price for 2000 sq.ft house:", predicted_price[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')

plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Price Prediction Based on Area")
plt.show()