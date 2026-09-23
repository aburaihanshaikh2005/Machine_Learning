from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#1.Load Data
data=load_breast_cancer()
x=data.data
y=data.target

#2.Train-Test Split(Stratified)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=0.2,random_state=42,stratify=y
)

#3.Feature Scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#4.Model Training
model=LogisticRegression(max_iter=1000,random_state=42)
model.fit(x_train,y_train)

#5.Prediction
y_pred=model.predict(x_test)

#Evaluation metrices
print("---Logistic Regression Performance---")
print(f"Accuracy:{accuracy_score(y_test,y_pred):.4f}")
print(f"Precision:{precision_score(y_test,y_pred):.4f}")
print(f"Recall:{recall_score(y_test,y_pred):.4f}")
print(f"F1 Score:{f1_score(y_test,y_pred):.4f}")
