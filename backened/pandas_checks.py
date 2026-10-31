"""Prints the correct answers (computed directly with Pandas) for the test questions.
Compare these with what POST /ask returns.   Run from backend/:  python pandas_checks.py
"""
import pandas as pd

from app import dataset_manager as dm

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

hr = dm.load_dataset("hr_attrition")
st = dm.load_dataset("students")


def show(title, value):
    print(f"\n--- {title}")
    print(value)


print("===== HR ATTRITION =====")
show("How many employees left the company?", (hr["Attrition"] == "Yes").sum())
show("How many employees work overtime?", (hr["OverTime"] == "Yes").sum())
show("How many employees earn more than 10000 per month?", (hr["MonthlyIncome"] > 10000).sum())
show("Average salary by department", hr.groupby("Department")["MonthlyIncome"].mean())
show("Average age by job role", hr.groupby("JobRole")["Age"].mean().sort_values(ascending=False))

print("\n\n===== STUDENTS =====")
show("Average math score, students who completed test prep",
     st.loc[st["test preparation course"] == "completed", "math score"].mean())
show("Students per parental level of education", st["parental level of education"].value_counts())
show("Average reading score by gender", st.groupby("gender")["reading score"].mean())
show("Students who scored above 90 in math", (st["math score"] > 90).sum())
