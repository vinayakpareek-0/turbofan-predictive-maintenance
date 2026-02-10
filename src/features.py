import pandas as pd
import numpy as np

def add_rolling_features(df, sensors, window=13):
    """Calculates rolling mean and standard deviation for specified sensors."""
    for s in sensors:
        # Calculate per unit_id to avoid leakage between different engines
        rolling = df.groupby('unit_id')[s].rolling(window=window, min_periods=1)
        df[f'{s}_mean'] = rolling.mean().reset_index(level=0, drop=True)
        df[f'{s}_std'] = rolling.std().reset_index(level=0, drop=True).fillna(0)
    return df

def add_lag_features(df, sensors, lag=1):
    """Calculates the change (delta) between the current and previous sensor state."""
    for s in sensors:
        df[f'{s}_delta'] = df.groupby('unit_id')[s].diff(periods=lag).fillna(0)
    return df

def apply_rul_clipping(df, dataset_id):
    """Applies piecewise linear clipping to the RUL target."""
    # 125 for stable sets (001/003), 150 for complex sets (002/004)
    clip_limit = 155 if dataset_id in ['FD002', 'FD004'] else 130
    df['RUL_clipped'] = df['RUL'].clip(upper=clip_limit)
    return df

def run_feature_engineering(df, dataset_id):
    """Orchestrates temporal feature extraction and target clipping."""
    # Identify sensors that are present in the dataframe
    sensors = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
    
    # 1. Temporal Dynamics (Windowing & Lags)
    df = add_rolling_features(df, sensors, window=10)
    df = add_lag_features(df, sensors)
    
    # 2. Final Target Engineering
    if 'RUL' in df.columns:
        df = apply_rul_clipping(df, dataset_id)
        
    return df