import pandas as pd
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def list_low_variance_sensors(df, dataset_id):
    """Detects low-variance sensors using specific CV and IQR thresholds."""
    thresholds = {
        'FD001': {'cv': 0.00002, 'iqr': 0.00005, 'manual': ['s6']},
        'FD002': {'cv': 0.015,   'iqr': 0.010,   'manual': []},
        'FD003': {'cv': 0.00005, 'iqr': 0.00005, 'manual': ['s6', 's10']},
        'FD004': {'cv': 0.015,   'iqr': 0.010,   'manual': []}
    }
    
    cfg = thresholds[dataset_id]
    sensors = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
    to_drop = []

    for s in sensors:
        mean, std, median = df[s].mean(), df[s].std(), df[s].median()
        iqr = df[s].quantile(0.75) - df[s].quantile(0.25)
        cv = std / abs(mean) if mean != 0 else 0
        iqr_ratio = iqr / abs(median) if median != 0 else 0
        
        if (cv < cfg['cv']) and (iqr_ratio < cfg['iqr']):
            to_drop.append(s)
            
    to_drop = sorted(list(set(to_drop + cfg['manual'])))
    print(f"❌ {dataset_id} Low-Var Candidates: {to_drop}")
    return to_drop

def list_noisy_sensors(df, dataset_id):
    """Detects sensors with low correlation to normalized engine age."""
    corr_thresholds = {'FD001': 0.2, 'FD002': 0.0025, 'FD003': 0.05, 'FD004': 0.0015}
    
    df_tmp = df.copy()
    max_time = df_tmp.groupby('unit_id')['time'].transform('max')
    df_tmp['norm_age'] = df_tmp['time'] / max_time
    
    sensors = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
    threshold = corr_thresholds[dataset_id]
    
    noisy = [s for s in sensors if abs(df_tmp[s].corr(df_tmp['norm_age'])) < threshold]
    print(f"🔇 {dataset_id} Noisy Candidates (|r| < {threshold}): {noisy}")
    return noisy

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def filter_by_config(df, dataset_id, config):
    selected_sensors = config['features']['remaining_sensors'].get(dataset_id, [])
    core_cols = ['unit_id', 'time', 'setting1', 'setting2', 'setting3']
    if 'RUL' in df.columns:
        core_cols.append('RUL')
    cols_to_keep = core_cols + [c for c in selected_sensors if c not in core_cols]
    return df[cols_to_keep]

def calculate_initial_rul(df):
    df['RUL'] = df.groupby('unit_id')['time'].transform('max') - df['time']
    return df

def apply_normalization(df, dataset_id, fitted_models=None):
    """
    Ensures test data is normalized using training parameters.
    fitted_models: dict containing {'kmeans': model, 'scalers': {regime_id: scaler}}
    """
    sensors = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
    settings = ['setting1', 'setting2', 'setting3']
    
    # CASE: FD002/FD004 (Regime-Specific)
    if dataset_id in ['FD002', 'FD004']:
        if fitted_models is None:
            # Training Mode: Fit new KMeans and Scalers
            kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
            df['regime_id'] = kmeans.fit_predict(df[settings])
            
            scalers = {}
            for rid in range(6):
                mask = df['regime_id'] == rid
                if mask.any():
                    scaler = StandardScaler()
                    df.loc[mask, sensors] = scaler.fit_transform(df.loc[mask, sensors])
                    scalers[rid] = scaler
            return df, {'kmeans': kmeans, 'scalers': scalers}
        else:
            # Test Mode: Use existing KMeans and Scalers
            df['regime_id'] = fitted_models['kmeans'].predict(df[settings])
            for rid, scaler in fitted_models['scalers'].items():
                mask = df['regime_id'] == rid
                if mask.any():
                    df.loc[mask, sensors] = scaler.transform(df.loc[mask, sensors])
            return df, fitted_models

    # CASE: FD001/FD003 (Global)
    else:
        if fitted_models is None:
            scaler = StandardScaler()
            df[sensors] = scaler.fit_transform(df[sensors])
            return df, {'scaler': scaler}
        else:
            df[sensors] = fitted_models['scaler'].transform(df[sensors])
            return df, fitted_models

def run_preprocessing_pipeline(df, dataset_id, config, fitted_models=None):
    df = calculate_initial_rul(df)
    df = filter_by_config(df, dataset_id, config)
    df, models = apply_normalization(df, dataset_id, fitted_models)
    return df, models

