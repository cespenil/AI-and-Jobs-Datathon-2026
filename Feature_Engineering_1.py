import pandas as pd
from sklearn.preprocessing import StandardScaler
from cleaning import df_clean

#print(df.head())
#print(df.info())
#print(df.describe())

df_numer = df_clean[['ai_risk_score', 'salary']]

# scaler object - class instance to normalize / standardize data
# transforms data to have a mean of 0 and a sd of 1
scaler = StandardScaler()

# fit scaler on the numeric columns and transform
df_numer_scal = scaler.fit_transform(df_numer)

#convert to dataframe (risk score vs salary)
df_numer_scal = pd.DataFrame(df_numer_scal, columns=['ai_risk_score', 'salary'])

# convert to dataframe (relative risk per salary unit)
df_numer_scal['risk_per_salary'] = df_numer_scal['ai_risk_score'] / (df_numer_scal['salary'] + 1e-6)

# multiplying features to capture relationships and covert 
# to dataframe
df_numer_scal['ai_risk_vs_salary'] = df_numer_scal['ai_risk_score'] * df_numer_scal['salary']

#check
# print(df_numer_scal.head())


