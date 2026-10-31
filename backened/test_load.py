from app import dataset_manager as dm

for d in ["superstore", "students", "hr_attrition"]:
    s = dm.get_schema(d)
    print(d, s["rows"], "rows,", s["column_count"], "columns")