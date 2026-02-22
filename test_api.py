import pandas as pd
import requests
import json
import os

def run_sequence_test(window_size=30, target_unit=1):
    url = "http://localhost:8000/predict"
    dataset_id = "fd004"
    
    # Paths
    processed_path = f"data/processed/test_{dataset_id.upper()}_final.csv"
    rul_truth_path = f"data/interim/RUL_{dataset_id.upper()}.csv"

    try:
        # 1. Load Data
        df = pd.read_csv(processed_path)
        # RUL file has one column 'RUL', where index 0 is Unit 1, index 1 is Unit 2...
        df_truth = pd.read_csv(rul_truth_path)

        # 2. Group by Unit and extract the sequence
        # We find our specific engine
        if target_unit not in df['unit_id'].unique():
            print(f"❌ Unit {target_unit} not found in dataset.")
            return

        unit_data = df[df['unit_id'] == target_unit].sort_values('time')
        
        if len(unit_data) < window_size:
            print(f"⚠️ Warning: Unit {target_unit} only has {len(unit_data)} cycles. Using full available history.")
            sequence_data = unit_data
        else:
            # Grab the last N cycles (the window)
            sequence_data = unit_data.tail(window_size)

        # 3. Get Ground Truth for this specific unit
        # Note: dataset units start at 1, but CSV index starts at 0
        actual_rul = df_truth.iloc[target_unit - 1]['RUL']

        # 4. Prepare Payload
        # We send everything except internal labels
        payload_sequence = sequence_data.drop(columns=['RUL', 'RUL_clipped'], errors='ignore').to_dict(orient='records')
        
        payload = {
            "dataset_id": dataset_id,
            "sequence": payload_sequence
        }

        print(f"📡 Sending window of {len(payload_sequence)} cycles for Engine Unit: {target_unit}")
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            predicted_rul = result['predicted_rul']
            
            print("\n" + "="*40)
            print(f"🛠️  Dataset: {dataset_id}, ENGINE UNIT {target_unit} REPORT")
            print("="*40)
            print(f"📊 ACTUAL RUL:    {actual_rul} cycles")
            print(f"🤖 PREDICTED RUL: {predicted_rul} cycles")
            
            error = round(float(predicted_rul) - float(actual_rul), 2)
            print(f"📉 ERROR:         {error} cycles")
            
            if abs(error) < 10:
                print("🎯 Status: Highly Accurate Prediction")
            elif error > 0:
                print("⚠️ Status: Overestimating Life (Dangerous)")
            else:
                print("✅ Status: Underestimating Life (Conservative/Safe)")
            print("="*40)
        else:
            print(f"❌ API Error: {response.text}")

    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    # Test for Unit 1 with a 30-cycle history
    run_sequence_test(window_size=30, target_unit=1)