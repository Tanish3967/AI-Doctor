import json
import pickle
import os

json_path = os.path.join(os.path.dirname(__file__), '../medicine_dataset.json')
pkl_path = os.path.join(os.path.dirname(__file__), '../medicine_dataset.pkl')

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(pkl_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Converted {json_path} to {pkl_path}.")
