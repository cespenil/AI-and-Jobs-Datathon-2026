import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from cleaning import df_clean

#primary skill into a numeric categorical representation
le_skill = LabelEncoder()

scaler = StandardScaler()

#convert to dataframe
df_skills = df_clean[['primary_skill', 'salary']].copy()
df_skills['primary_skill_encoded'] = le_skill.fit_transform(df_skills['primary_skill'])

#skill popularity
skill_counts = df_skills['primary_skill'].value_counts()
df_skills['skill_popularity'] = df_skills['primary_skill'].map(skill_counts)

#skill vs salary ratio / risk_to_salary before
df_skills['skill_salary_ratio'] = df_skills['primary_skill_encoded'] / (df_skills['salary'] + 1e-6)

#interaction: multiply encoded skill by salary
df_skills['skill_vs_salary'] = df_skills['primary_skill_encoded'] * df_skills['salary']

#normalize salary and derived features
df_skills[['salary', 'skill_salary_ratio', 'skill_vs_salary']] = scaler.fit_transform(
    df_skills[['salary', 'skill_salary_ratio', 'skill_vs_salary']]
)

df_skills_final = df_skills
