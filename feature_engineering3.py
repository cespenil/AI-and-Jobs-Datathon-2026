# feature_engineering3.py

import pandas as pd
from cleaning import df

# copy of the main dataframe
df_FE = df.copy()

# cleaning salary
salary_data = df_FE["salary"]
print("Salary data type: ", salary_data.dtype)
print("Initial rows of salary data", salary_data.head())

if salary_data.dtype == "object":
    df_FE["salary_cleaned"] = pd.to_numeric(
        df_FE["salary"].replace(r"[$,]", "", regex=True),
        errors="coerce"
    )
else:
    print("Salary is numeric with a little to no fixes")
    df_FE["salary_cleaned"] = salary_data

# basic validation
print("Zeros: ", (df_FE["salary_cleaned"] == 0).sum())
print("Negatives: ", (df_FE["salary_cleaned"] < 0).sum())
print(df_FE["salary_cleaned"].describe())

# Year cleaning
df_FE["year"] = pd.to_numeric(df_FE["year"], errors="coerce")
df_FE["year_from_start"] = df_FE["year"] - 2015

# rolling average
df_FE = df_FE.sort_values("year")
# new column of the mean salary that was cleaned
df_FE["salary_rolling_average"] = df_FE["salary_cleaned"].rolling(3).mean()

# final cleanup
df_FE = df_FE.dropna(subset=["salary_cleaned", "year", "year_from_start"])

# temproary testing
print(df_FE.columns)
print()
print(df_FE[["salary_cleaned", "year", "year_from_start"]].dtypes)
print()
print(df_FE[["salary_cleaned", "year", "year_from_start"]].isnull().sum())
print()
print(df_FE[["year", "year_from_start", "salary_cleaned"]].tail())
print()
print(df_FE.shape)