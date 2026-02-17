import os
import torch
import pandas as pd
import numpy as np
from src.data_loader import convert_all_raw_data
from src.preprocessing import run_preprocessing_pipeline, load_config
from src.features import run_feature_engineering

# Model-specific imports
from src.modeling import train_model, evaluate_on_test
from src.deep_learning import RUL_LSTM, prepare_lstm_data, train_model_dl, evaluate_lstm

def main():
    
    # GLOBAL CONFIGURATION
    # Switch between "XGB" and "LSTM" here
    MODEL_MODE = "LSTM" 
    
    # LSTM Specifics (Only used if MODE is LSTM)
    WINDOW_SIZE = 30
    BATCH_SIZE = 64
    EPOCHS = 50
    
    raw_dir, interim_dir, processed_dir = "data/raw", "data/interim", "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_results = []

    print(f"🚀 Step 1: Data Initialization (Mode: {MODEL_MODE})")
    convert_all_raw_data(raw_dir, interim_dir)
    
    print(f"\n🚀 Step 2: Executing {MODEL_MODE} Pipeline...")
    
    for ds in datasets:
        print(f"\n📦 Processing Dataset: {ds}")
        
        # Load Data
        train_df = pd.read_csv(f"{interim_dir}/train_{ds}.csv")
        test_df = pd.read_csv(f"{interim_dir}/test_{ds}.csv")
        y_truth = pd.read_csv(f"{interim_dir}/RUL_{ds}.csv")['RUL'].values
        
        # A. Preprocessing (Regime Clustering & Scaling)
        train_proc, fitted_models = run_preprocessing_pipeline(train_df, ds, config)
        test_proc, _ = run_preprocessing_pipeline(test_df, ds, config, fitted_models=fitted_models)

        # B. Feature Engineering (Rolling Stats & Trends)
        # We use the processed features for both models to ensure a fair comparison
        train_final = run_feature_engineering(train_proc, ds)
        test_final = run_feature_engineering(test_proc, ds)
                
        # Drop IDs and target columns for training
        drop_cols = ['unit_id', 'time', 'regime_id', 'RUL', 'RUL_clipped']
        features = [c for c in train_final.columns if c not in drop_cols]
        
        # 2. MODELING BRANCH
        if MODEL_MODE == "XGB":
            print(f"   Training XGBoost Regressor...")
            model = train_model(train_final[features], train_final['RUL_clipped'], ds)
            rmse, score, _ = evaluate_on_test(model, test_final, y_truth, features)
            
        elif MODEL_MODE == "LSTM":
            print(f"   Generating {WINDOW_SIZE}-cycle sequences for LSTM...")
            X_train, y_train = prepare_lstm_data(train_final, WINDOW_SIZE, features, 'RUL_clipped')
            
            print(f"   Training LSTM on {device}...")
            model = RUL_LSTM(input_dim=len(features)).to(device)
            model = train_model_dl(X_train, y_train, model, batch_size=BATCH_SIZE, epochs=EPOCHS)
            
            # Predict using the terminal sequence of each engine
            rmse, score = evaluate_lstm(model, test_final, y_truth, features, WINDOW_SIZE)
        
        all_results.append({
            'Dataset': ds, 
            'Mode': MODEL_MODE,
            'RMSE': round(rmse, 2), 
            'NASA Score': round(score, 2)
        })
        print(f"   ✅ {ds} Metrics -> RMSE: {rmse:.2f} | NASA Score: {score:.2f}")

    summary_df = pd.DataFrame(all_results)
    print("\n")
    print(f"           TURBOFAN PIPELINE SUMMARY ({MODEL_MODE})")
    print("."*10)
    print(summary_df.to_string(index=False))
    print("."*10)

if __name__ == "__main__":
    main()