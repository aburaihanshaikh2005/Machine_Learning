import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

df["area"] = df["rm"] * 500
df["bedrooms"] = (df["rm"] / 2).round().astype(int)

X = df[["area", "bedrooms"]]
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)
poly_r2 = r2_score(y_test, poly_pred)

print("--- R2 Score Comparison ---")
print("Linear Regression R2 Score:", linear_r2)
print("Polynomial Regression R2 Score:", poly_r2)

print("\n--- Comparison ---")
if poly_r2 > linear_r2:
    print("Polynomial Regression has a higher R2 score.")
elif poly_r2 < linear_r2:
    print("Linear Regression has a higher R2 score.")
else:
    print("Both models have the same R2 score.")