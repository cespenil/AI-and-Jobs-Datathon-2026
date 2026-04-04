import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("future_jobs_ai_data.csv")


df.info()
df.head()

#data cleaning here
df = df.fillna(0)  #fills missing values with 0
df = df.dropna()   #drops rows with missing values
df = df.drop_duplicates()  #removes duplicate rows
#print(df.isnull().sum())  #checks for remaining missing valuess
#df = df.dropna()  drops rows with missing values (already have df.fillna(0) so I don't really need this)

print(df.describe())  #provides summary statistics of the dataset
#df["job_title"].unique() #checks what values are in job_title (categorical variable)
#print(df["country"].unique()) #checks what values are in country (categorical variable)
print(df.dtypes)  #checks data types of each column
#print(df["salary"].min())
#print(df["salary"].max())
#print(df["ai_risk_score"].describe())   this one is not based on 0 or 1 but has range of values such as mean, min, ect

#working on the columns here
#df.columns = df.columns.str.strip()  everything is good here for both
#df.columns = df.columns.str.lower()

