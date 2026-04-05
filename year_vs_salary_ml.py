# year_vs_salary_ml.py

# all imports
from feature_engineering3 import df_FE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# X data and Y data
X = df_FE[["year_from_start"]]
y = df_FE["salary_cleaned"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("The mean squared error of the cleaned salary training data:")
print(mean_squared_error(y_train, y_train_pred))

print("The mean squared error of the cleaned salary testing data:")
print(mean_squared_error(y_test, y_test_pred))

print("The mean absolute error of the cleaned salary trained data:")
print(mean_absolute_error(y_train, y_train_pred))

print("The mean absolute error of the cleaned salary testing data:")
print(mean_absolute_error(y_test, y_test_pred))

print("The r2 score of the cleaned salary training data:")
print(r2_score(y_train, y_train_pred))

print("The r2 score of the cleaned salary testing data:")
print(r2_score(y_test, y_test_pred))
