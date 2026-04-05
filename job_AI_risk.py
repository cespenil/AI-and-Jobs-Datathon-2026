# job_AI_risk.py

from cleaning import df
import pandas as pd


# Aggregate by job_title
job_summary = df.groupby('job_title').agg({
    'ai_risk_score': 'mean',        # Average AI risk
    'skill_demand_score': 'mean',   # Average skill demand
    'job_openings': 'sum',          # Total openings
    'salary': 'mean'                # Average salary
}).reset_index()


# Future score used for Qualitative AI risk
job_summary['future_score'] = job_summary['skill_demand_score'] * (1 - job_summary['ai_risk_score'])


# Categorize AI risk
def risk_category(risk):
    if risk < 0.3:
        return 'Low Risk'
    elif risk < 0.6:
        return 'Medium Risk'
    else:
        return 'High Risk'

job_summary['risk_category'] = job_summary['ai_risk_score'].apply(risk_category)


# Risk-adjusted job openings (numeric feature)
# Basically a feature that calculates the amount of "safe" jobs based on the
# openings and AI risk score
job_summary['risk_adjusted_openings'] = job_summary['job_openings'] * (1 - job_summary['ai_risk_score'])
# Scale risk-adjusted openings for easier display (In the millions)
job_summary['risk_adjusted_openings_millions'] = job_summary['risk_adjusted_openings'] / 1e6


# Create demand category for quadrants
median_safe_jobs = job_summary['risk_adjusted_openings_millions'].median()
job_summary['demand_category'] = job_summary['risk_adjusted_openings_millions'].apply(
    lambda x: 'High Demand' if x >= median_safe_jobs else 'Low Demand'
)


# Combine risk + demand into quadrants
job_summary['quadrant'] = job_summary['risk_category'] + " / " + job_summary['demand_category']


# Ranking by safety
job_summary['rank_by_safety'] = job_summary['risk_adjusted_openings'].rank(ascending=False)


# Normalized safety score (0-1) for visualization
job_summary['safety_score_norm'] = (
    job_summary['risk_adjusted_openings'] - job_summary['risk_adjusted_openings'].min()
) / (
    job_summary['risk_adjusted_openings'].max() - job_summary['risk_adjusted_openings'].min()
)


# Creating a csv file to use in Omni
job_summary.to_csv("job_summary_featured.csv", index=False)

# Checking dataframe
print(job_summary.head())
