'''Modify the decision threshold explicitly in Python to 0.3, 0.5, and 0.7
Compare how the Precision and Recall metrics change.'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

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
    x, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(x_train, y_train)

y_probability = model.predict_proba(x_test)[:, 1]

thresholds = [0.3, 0.5, 0.7]

print("\nThreshold Comparison")
print("-------------------------------")
print("Threshold   Precision   Recall")
print("-------------------------------")

for threshold in thresholds:
    y_pred = (y_probability >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"{threshold:<11} {precision:.2f}        {recall:.2f}")