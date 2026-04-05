from cleaning import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#define safe features (independent of target)
features = ['salary']
X = df[features]
y = df['ai_risk_score']

#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train model
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")
    if name == "RandomForest":
        rf_model = model
        rf_pred = y_pred 

if __name__ == '__main__':
    print(f"Mean Squared Error: {mse}")
    print(f"R2 score: {r2}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
    plt.xlabel("Actual AI Risk Score")
    plt.ylabel("Predicted AI Risk Score")
    plt.title("Random Forest Predictions")
    plt.show()