# Activity 3
# Car Price Prediction using Linear and Polynomial Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("activity 3/Car Price Prediction.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# Select Features and Target
X = df[[
    "horsepower",
    "enginesize",
    "curbweight",
    "citympg"
]]

y = df["price"]


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Linear Regression Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression")
print("RMSE:", rmse)
print("R2 Score:", r2)


# Polynomial Regression
degrees = [2, 3, 4]

for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()

    poly_model.fit(X_train_poly, y_train)

    y_poly_pred = poly_model.predict(X_test_poly)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_poly_pred)
    )

    r2 = r2_score(
        y_test,
        y_poly_pred
    )

    print("\nPolynomial Regression Degree:", degree)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# Plot Polynomial Regression
# Using Horsepower for the x-axis

X_plot = df[["horsepower"]]
y_plot = df["price"]


plt.scatter(
    X_plot,
    y_plot,
    label="Actual Data"
)


for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X_plot)

    poly_model = LinearRegression()

    poly_model.fit(X_poly, y_plot)

    y_poly = poly_model.predict(X_poly)

    plt.plot(
        X_plot,
        y_poly,
        label="Degree " + str(degree)
    )


plt.xlabel("Horsepower")
plt.ylabel("Car Price")
plt.title("Polynomial Regression - Car Price Prediction")

plt.legend()
plt.show()