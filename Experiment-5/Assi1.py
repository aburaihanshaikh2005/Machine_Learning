''' Build a Logistic Regression model  for a custom student dataset to predict pass/fail
    based on Study Hours and Attendence.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Study_Hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9],
    "Attendance": [45, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80, 85, 90],
    "Pass": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

x = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

study_hours = float(input("\nEnter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))

prediction = model.predict([[study_hours, attendance]])

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")