import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

df["area"] = df["rm"] * 500
df["bedrooms"] = (df["rm"] / 2).round().astype(int)

print("--- Dataset ---")
print(df[["area", "bedrooms", "medv"]].head())

X = df[["area", "bedrooms"]]
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Information ---")
print("Area Coefficient:", model.coef_[0])
print("Bedroom Coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

new_house = pd.DataFrame({
    "area": [3000],
    "bedrooms": [3]
})

predicted_price = model.predict(new_house)

print("\n--- New House Prediction ---")
print("House Area:", new_house["area"][0], "sq. ft.")
print("Bedrooms:", new_house["bedrooms"][0])
print("Predicted House Price: $", round(predicted_price[0] * 1000, 2))
