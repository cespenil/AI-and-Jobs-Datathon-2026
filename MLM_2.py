# MLM 2 - Skills vs Salary
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#importing the feature-enineered dataframe
from Feature_Engineering_2 import df_skills_final

#select features and target
X = df_skills_final[['primary_skill_encoded', 'skill_popularity', 'skill_salary_ratio', 'skill_vs_salary']]
y = df_skills_final['salary']

# create a Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#fit the model on the training data
lr_model = LinearRegression()

#learned patters from the training data
lr_model.fit(X_train, y_train)

#prediction based of the previous data
y_pred = lr_model.predict(X_test)

#measures prediction error, lower the better
mse = mean_squared_error(y_test, y_pred)

#the R^2 score shows how much variance the model explains
r2 = r2_score(y_test, y_pred)

if __name__ == '__main__':
    print('Mean Squared Error:', mse)
    print('R2 Score:', r2)



