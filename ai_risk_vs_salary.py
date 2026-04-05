from cleaning import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
df_numeric_features = df[['ai_risk_score', 'salary']]

#scale features
scaler = StandardScaler()
df_numeric_scaled = scaler.fit_transform(df_numeric_features)
df_numeric_scaled = pd.DataFrame(df_numeric_scaled, columns=['ai_risk_score', 'salary'])

#create new features 
df_numeric_scaled['risk_per_salary'] = df_numeric_scaled['ai_risk_score'] / (df_numeric_scaled['salary'] + 1e-6)
df_numeric_scaled['ai_risk_vs_salary'] = df_numeric_scaled['ai_risk_score'] * df_numeric_scaled['salary']

# Define features and target variable
X = df_numeric_scaled[['salary', 'risk_per_salary', 'ai_risk_vs_salary']]  # exclude ai_risk_score here
y = df_numeric_scaled['ai_risk_score']  # target remains ai_risk_score 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

'''

# 1. Correlation between AI risk score and salary
corr_salary = df['ai_risk_score'].corr(df['salary'])
print(f"Correlation between AI risk score and salary: {corr_salary:.2f}")

# 2. Scatter plot: AI risk score vs Salary
plt.scatter(df['salary'], df['ai_risk_score'], alpha=0.5)
plt.xlabel('Salary')
plt.ylabel('AI Risk Score')
plt.title('AI Risk Score vs Salary')
plt.show()

# 3. Correlations of AI risk score with other numeric variables
numeric_cols = ['salary', 'skill_demand_score', 'job_openings', 'year', 'job_survival_class']
corrs = df[numeric_cols + ['ai_risk_score']].corr()['ai_risk_score'].drop('ai_risk_score')
print("\nCorrelation of AI risk score with other numeric variables:")
print(corrs)

# 4. Average AI risk score by experience level (categorical example)
avg_risk_by_exp = df.groupby('experience_level')['ai_risk_score'].mean()
print("\nAverage AI risk score by experience level:")
print(avg_risk_by_exp)

# 5. (Optional) Bar plot for categorical variable
avg_risk_by_exp.plot(kind='bar', title='Average AI Risk Score by Experience Level')
plt.ylabel('Average AI Risk Score')
plt.show()