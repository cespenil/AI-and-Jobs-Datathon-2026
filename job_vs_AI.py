from cleaning import df
import pandas as pd

#print(df.head())
#print(df.columns)

job_summary = df.groupby('job_title').agg({
    'ai_risk_score': 'mean',        # Y-axis for scatterplot
    'skill_demand_score': 'mean',   # X-axis for scatterplot
    'job_openings': 'sum',          # Bubble size
    'salary': 'mean'                # Optional: bubble color
}).reset_index()

# Step 2: Compute a 'future_score' metric (optional, helps highlight safe vs risky jobs)
job_summary['future_score'] = job_summary['skill_demand_score'] * (1 - job_summary['ai_risk_score'])

# Step 3: Categorize AI risk for storytelling / visualization
def risk_category(risk):
    if risk < 0.3:
        return 'Low Risk'
    elif risk < 0.6:
        return 'Medium Risk'
    else:
        return 'High Risk'

job_summary['risk_category'] = job_summary['ai_risk_score'].apply(risk_category)