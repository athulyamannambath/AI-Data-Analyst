from app import dataset_manager as dm
from app.engine import run_plan

df = dm.load_dataset("superstore")
print(run_plan(df, {"operation": "groupby_aggregate", "group_by": "Region",
                    "column": "Sales", "aggregation": "sum", "sort": "desc"}))

hr = dm.load_dataset("hr_attrition")
print(run_plan(hr, {"operation": "groupby_aggregate", "group_by": "Department",
                    "column": "MonthlyIncome", "aggregation": "mean"}))