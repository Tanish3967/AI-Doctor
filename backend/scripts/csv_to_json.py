import pandas as pd
import json
import os

csv_path = os.path.join(os.path.dirname(__file__), '../medicine_dataset.csv')
json_path = os.path.join(os.path.dirname(__file__), '../medicine_dataset.json')

df = pd.read_csv(csv_path, low_memory=False)
sideeffect_cols = [f'sideeffect{i}' for i in range(1, 27)]

drug_data = {}
for _, row in df.iterrows():
    med_name = str(row.get('medicine_name', '')).strip().lower()
    if not med_name:
        continue
    side_effects = [str(row.get(col, '')).strip() for col in sideeffect_cols if pd.notna(row.get(col, '')) and str(row.get(col, '')).strip()]
    med_dict = row.to_dict()
    med_dict['side_effects'] = side_effects
    drug_data[med_name] = med_dict

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(drug_data, f, ensure_ascii=False, indent=2)
print(f"Converted {csv_path} to {json_path}.")
