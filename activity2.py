# Activity 2
# Telco Customer Churn Prediction using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# Load Dataset
df = pd.read_csv("activity 2/telco.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# Clean Dataset
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])


# Convert Churn into numbers
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# Select Features and Target
X = df[[
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService"
]]

y = df["Churn"]


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Numerical and Categorical Features
numerical_features = [
    "tenure",
    "MonthlyCharges"
]

categorical_features = [
    "Contract",
    "InternetService"
]


# Encoding and Scaling
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])


# Transform Training and Testing Data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()