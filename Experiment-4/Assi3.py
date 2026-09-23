'''3. Implement Polynomial Regression on the same dataset and compare the R2
score against standard Linear Regression.'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = pd.read_csv("house_prices_dataset.csv")

X = data[['square_feet']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_pred)

print("Linear Regression R2 Score:", linear_r2)
print("Polynomial Regression R2 Score:", poly_r2)

if poly_r2 > linear_r2:
    print("Polynomial Regression has a higher R2 score.")
elif poly_r2 < linear_r2:
    print("Linear Regression has a higher R2 score.")
else:
    print("Both models have the same R2 score.")

plt.scatter(X, y)
plt.plot(X, linear_model.predict(X), label="Linear Regression")
plt.plot(X, poly_model.predict(poly.transform(X)), label="Polynomial Regression")

plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()