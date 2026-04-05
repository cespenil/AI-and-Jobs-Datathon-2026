# ml_example.py

from cleaning import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Aggregate by job_title (same as your feature engineering)
job_summary = df.groupby('job_title').agg({
    'ai_risk_score': 'mean',
    'skill_demand_score': 'mean',
    'job_openings': 'sum',
    'salary': 'mean'
}).reset_index()

# Compute risk-adjusted openings
job_summary['risk_adjusted_openings'] = job_summary['job_openings'] * (1 - job_summary['ai_risk_score'])

# Select features and target
X = job_summary[['ai_risk_score', 'skill_demand_score', 'salary']]  # features
job_summary['risk_adjusted_openings_millions'] = job_summary['risk_adjusted_openings'] / 1e6
y = job_summary['risk_adjusted_openings_millions']  # target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results")
print("------------------------")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Check feature coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients)