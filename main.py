
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader   import convert_all_raw_data
from src.preprocessing import run_preprocessing_pipeline, load_config
from src.features      import run_feature_engineering

# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Turbofan RUL Prediction Pipeline")
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgb', 'lstm'],
        required=True,
        help="Choose model: 'xgb' or 'lstm'"
    )
    return parser.parse_args()


# LSTM CLASSES & FUNCTIONS

class NASALoss(nn.Module):
    def __init__(self):
        super(NASALoss, self).__init__()

    def forward(self, y_pred, y_true):
        d = y_pred.view(-1) - y_true.view(-1)
        loss = torch.where(
            d < 0,
            torch.exp(-d / 13.0) - 1.0,
            torch.exp(d  / 10.0) - 1.0
        )
        return torch.mean(loss)


class RUL_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(RUL_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.regressor(out[:, -1, :])


def prepare_lstm_data(df, window_size, feature_cols, target_col):
    X, y = [], []
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        if len(unit_data) >= window_size:
            data   = unit_data[feature_cols].values
            target = unit_data[target_col].values
            for i in range(len(unit_data) - window_size + 1):
                X.append(data[i:i + window_size])
                y.append(target[i + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model_dl(X_train, y_train, model, batch_size=64, epochs=50, lr=0.001):
    # ✅ Device fix: tensors follow model's device
    device    = next(model.parameters()).device
    X_tensor  = torch.from_numpy(X_train).to(device)
    y_tensor  = torch.from_numpy(y_train).to(device)

    dataset   = TensorDataset(X_tensor, y_tensor)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = NASALoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x).squeeze()
            loss   = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
    return model


def evaluate_lstm(model, test_df, y_truth, feature_cols, window_size):
    # ✅ Device fix: input tensor follows model's device
    device = next(model.parameters()).device
    model.eval()
    y_preds = []

    for unit_id in test_df['unit_id'].unique():
        unit_data   = test_df[test_df['unit_id'] == unit_id]
        last_window = unit_data[feature_cols].values[-window_size:]

        # Pad if engine has fewer cycles than window_size
        if len(last_window) < window_size:
            pad         = np.zeros((window_size - len(last_window), len(feature_cols)))
            last_window = np.vstack([pad, last_window])

        x_input = torch.from_numpy(last_window).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_input).item()
            y_preds.append(max(0, pred))

    y_preds    = np.array(y_preds)
    rmse       = np.sqrt(((y_preds - y_truth) ** 2).mean())
    diff       = y_preds - y_truth
    nasa_score = np.sum(np.where(diff < 0,
                                 np.exp(-diff / 13) - 1,
                                 np.exp(diff  / 10) - 1))
    return rmse, nasa_score

AVG_LIVES = {'FD001': 206, 'FD002': 206, 'FD003': 247, 'FD004': 245}

LSTM_CFG = {
    'FD001': {'batch': 64,  'hidden': 64,  'epochs': 50,  'lr': 0.001,  'dropout': 0.2},
    'FD002': {'batch': 32,  'hidden': 128, 'epochs': 75,  'lr': 0.0005, 'dropout': 0.3},
    'FD003': {'batch': 64,  'hidden': 64,  'epochs': 50,  'lr': 0.001,  'dropout': 0.2},
    'FD004': {'batch': 32,  'hidden': 128, 'epochs': 75,  'lr': 0.0005, 'dropout': 0.3},
}

# Window = 20% of average engine life (min 30)
WINDOW_SIZES = {ds: max(30, int(AVG_LIVES[ds] * 0.20)) for ds in AVG_LIVES}

def run_xgb(datasets, interim_dir, config):
    from src.modeling import train_model, evaluate_on_test
    all_results = []

    for ds in datasets:
        print(f"\n{'='*45}")
        print(f"  📦 Dataset: {ds}")
        print(f"{'='*45}")

        train_df = pd.read_csv(f"{interim_dir}/train_{ds}.csv")
        test_df  = pd.read_csv(f"{interim_dir}/test_{ds}.csv")
        y_truth  = pd.read_csv(f"{interim_dir}/RUL_{ds}.csv")['RUL'].values

        train_proc, fitted_models = run_preprocessing_pipeline(train_df, ds, config, fitted_models=None)
        test_proc, _              = run_preprocessing_pipeline(test_df,  ds, config, fitted_models=fitted_models)

        train_final = run_feature_engineering(train_proc, ds)
        test_final  = run_feature_engineering(test_proc,  ds)

        drop_cols = ['unit_id', 'time', 'regime_id', 'RUL', 'RUL_clipped']
        features  = [c for c in train_final.columns if c not in drop_cols]
        print(f"  📊 Features: {len(features)}")

        print(f"  🌲 Training XGBoost...")
        model          = train_model(train_final[features], train_final['RUL_clipped'], ds)
        rmse, score, _ = evaluate_on_test(model, test_final, y_truth, features)

        all_results.append({'Dataset': ds, 'RMSE': round(rmse, 2), 'NASA Score': round(score, 2)})
        print(f"  ✅ {ds} | RMSE: {rmse:.2f} | NASA Score: {score:.2f}")

    return all_results

def run_lstm(datasets, interim_dir, config, device):
    all_results = []

    print(f"\n📊 Smart Window Sizes:")
    for ds in datasets:
        cfg = LSTM_CFG[ds]
        print(f"   {ds}: window={WINDOW_SIZES[ds]} | batch={cfg['batch']} | "
              f"hidden={cfg['hidden']} | epochs={cfg['epochs']} | "
              f"lr={cfg['lr']} | dropout={cfg['dropout']}")

    for ds in datasets:
        print(f"\n{'='*45}")
        print(f"  📦 Dataset: {ds}")
        print(f"{'='*45}")

        cfg         = LSTM_CFG[ds]
        window_size = WINDOW_SIZES[ds]

        train_df = pd.read_csv(f"{interim_dir}/train_{ds}.csv")
        test_df  = pd.read_csv(f"{interim_dir}/test_{ds}.csv")
        y_truth  = pd.read_csv(f"{interim_dir}/RUL_{ds}.csv")['RUL'].values

        train_proc, fitted_models = run_preprocessing_pipeline(train_df, ds, config, fitted_models=None)
        test_proc, _              = run_preprocessing_pipeline(test_df,  ds, config, fitted_models=fitted_models)

        train_final = run_feature_engineering(train_proc, ds)
        test_final  = run_feature_engineering(test_proc,  ds)

        drop_cols = ['unit_id', 'time', 'regime_id', 'RUL', 'RUL_clipped']
        features  = [c for c in train_final.columns if c not in drop_cols]
        print(f"  📊 Features: {len(features)} | Window: {window_size}")

        print(f"  🔄 Generating sequences...")
        X_train, y_train = prepare_lstm_data(train_final, window_size, features, 'RUL_clipped')
        print(f"  ✅ X: {X_train.shape} | y: {y_train.shape}")

        print(f"  🧠 Training LSTM on {device}...")
        model = RUL_LSTM(
            input_dim  = len(features),
            hidden_dim = cfg['hidden'],
            dropout    = cfg['dropout']
        ).to(device)

        model = train_model_dl(
            X_train, y_train, model,
            batch_size = cfg['batch'],
            epochs     = cfg['epochs'],
            lr         = cfg['lr']
        )

        rmse, score = evaluate_lstm(model, test_final, y_truth, features, window_size)

        all_results.append({'Dataset': ds, 'RMSE': round(rmse, 2), 'NASA Score': round(score, 2)})
        print(f"\n  ✅ {ds} | RMSE: {rmse:.2f} | NASA Score: {score:.2f}")

    return all_results



def main():
    args        = parse_args()
    config      = load_config()
    datasets    = ['FD001', 'FD002', 'FD003', 'FD004']
    raw_dir     = "data/raw"
    interim_dir = "data/interim"
    os.makedirs("data/processed", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Step 1: Data Initialization")
    print(f"   Model  : {args.model.upper()}")
    print(f"   Device : {device}")
    convert_all_raw_data(raw_dir, interim_dir)

    print(f"\n🚀 Step 2: Running {args.model.upper()} Pipeline...")

    if args.model == 'xgb':
        results = run_xgb(datasets, interim_dir, config)
    elif args.model == 'lstm':
        results = run_lstm(datasets, interim_dir, config, device)

    # Summary
    summary_df = pd.DataFrame(results)
    print("\n" + "="*45)
    print(f"     TURBOFAN {args.model.upper()} SUMMARY")
    print("="*45)
    print(summary_df.to_string(index=False))
    print("="*45)


if __name__ == "__main__":
    main()

# Use and throw scores running:
#   python main.py --model xgb
#   python main.py --model lstm
