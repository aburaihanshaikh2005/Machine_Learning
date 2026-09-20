import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://gist.githubusercontent.com/jiisanatNSUT/ffa89d9f0359bfa00f29a9c450bdaa17/raw/homeprices.csv"

df = pd.read_csv(url)

print("--- Dataset ---")
print(df)

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Information ---")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


new_house = pd.DataFrame({"area": [3000]})
predicted_price = model.predict(new_house)

print("\n--- New House Prediction ---")
print("House Area:", new_house["area"][0], "sq. ft.")
print("Predicted House Price: $", round(predicted_price[0], 2))

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("House Price Prediction using Linear Regression")
plt.show()