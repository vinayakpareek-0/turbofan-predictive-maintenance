import os
import yaml
import pandas as pd
from src.data_loader import convert_all_raw_data
from src.preprocessing import run_preprocessing_pipeline, load_config
from src.features import run_feature_engineering
from src.modeling import train_model, evaluate_on_test

def main():
    # 1. Path Configuration
    raw_dir = "data/raw"
    interim_dir = "data/interim"
    processed_dir = "data/processed"
    
    # Ensure the processed directory exists before we try to save to it
    os.makedirs(processed_dir, exist_ok=True)
    
    print("🚀 Step 1: Converting Raw TXT to Interim CSV...")
    convert_all_raw_data(raw_dir, interim_dir)
    
    config = load_config()
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_results = []

    print("\n🚀 Step 2: Running End-to-End Pipeline...")
    
    for ds in datasets:
        print(f"\n📦 Processing Dataset: {ds}")
        
        # Load from Interim (Created in Step 1)
        train_df = pd.read_csv(f"{interim_dir}/train_{ds}.csv")
        test_df = pd.read_csv(f"{interim_dir}/test_{ds}.csv")
        y_truth = pd.read_csv(f"{interim_dir}/RUL_{ds}.csv")['RUL'].values
        
        # # A. Preprocessing (In-Memory)
        # train_proc = run_preprocessing_pipeline(train_df, ds, config)
        # test_proc = run_preprocessing_pipeline(test_df, ds, config)
        
        # # B. Feature Engineering (In-Memory)
        # train_final = run_feature_engineering(train_proc, ds)
        # test_final = run_feature_engineering(test_proc, ds)
        
        # # C. Save Final Features (Populates your processed folder)
        # train_final.to_csv(f"{processed_dir}/train_{ds}_final.csv", index=False)
        # test_final.to_csv(f"{processed_dir}/test_{ds}_final.csv", index=False)
        # print(f"   💾 Saved final features to {processed_dir}")

        # A. Process Training Data (Fits the models)
        train_proc, fitted_models = run_preprocessing_pipeline(train_df, ds, config, fitted_models=None)

        # B. Process Test Data (Uses training models)
        test_proc, _ = run_preprocessing_pipeline(test_df, ds, config, fitted_models=fitted_models)

        # C. Feature Engineering (Proceed as normal)
        train_final = run_feature_engineering(train_proc, ds)
        test_final = run_feature_engineering(test_proc, ds)
                
        # D. Modeling
        drop_cols = ['unit_id', 'time', 'regime_id', 'RUL', 'RUL_clipped']
        features = [c for c in train_final.columns if c not in drop_cols]
        
        model = train_model(train_final[features], train_final['RUL_clipped'], ds)
        
        # E. Evaluation
        rmse, score, y_pred = evaluate_on_test(model, test_final, y_truth, features)
        
        all_results.append({
            'Dataset': ds, 'RMSE': round(rmse, 2), 'NASA Score': round(score, 2)
        })
        print(f"   ✅ {ds} Complete | RMSE: {rmse:.2f} | Score: {score:.2f}")

    # Final Summary
    summary_df = pd.DataFrame(all_results)
    print("\n" + "="*45)
    print("         TURBOFAN PROJECT SUMMARY")
    print("="*45)
    print(summary_df.to_string(index=False))
    print("="*45)

if __name__ == "__main__":
    main()