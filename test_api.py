import pandas as pd
import requests
import json
import os

def run_test():
    url = "http://localhost:8000/predict"
    dataset_id = "fd003"
    test_file = f"data/processed/test_{dataset_id.upper()}_final.csv"

    if not os.path.exists(test_file):
        print(f"❌ Error: {test_file} not found.")
        return

    try:
        df = pd.read_csv(test_file)
        
        # 1. Select a row (e.g., the first one)
        row = df.iloc[0]
        
        # 2. Extract the ORIGINAL RUL (Ground Truth)
        # We check for 'RUL' or 'RUL_clipped' depending on your pipeline
        actual_rul = row.get('RUL', "N/A")
        
        # 3. Prepare the features (dropping IDs and target)
        drop_cols = ['unit_id', 'time', 'regime_id', 'RUL', 'RUL_clipped']
        sample_dict = row.drop(labels=drop_cols, errors='ignore').to_dict()
        
        payload = {
            "dataset_id": dataset_id,
            "features": sample_dict
        }

        print(f"📡 Sending request for Unit {int(row['unit_id'])} at Cycle {int(row['time'])}...")
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            predicted_rul = result['predicted_rul']
            
            # 4. Compare the results
            print("\n✅ Prediction Successful!")
            print("=" * 40)
            print(f"📊 ACTUAL RUL:    {actual_rul} cycles")
            print(f"🤖 PREDICTED RUL: {predicted_rul} cycles")
            
            # Calculate Error
            error = round(float(predicted_rul) - float(actual_rul), 2)
            print(f"📉 ERROR:         {error} cycles")
            print("=" * 40)
            
            if result['warning']:
                print("⚠️  MAINTENANCE WARNING: Engine nearing failure!")
        else:
            print(f"❌ API Error: {response.text}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_test()