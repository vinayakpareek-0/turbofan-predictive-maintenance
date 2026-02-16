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

def create_sequences(df, window_size, feature_cols, target_col=None):
    """
    Transforms 2D dataframe into 3D sequences for Deep Learning.
    Shape: (num_samples, window_size, num_features)
    """
    sequences = []
    targets = []
    
    # Process each engine unit separately to avoid cross-engine sequences
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        
        # We can only create a sequence if the engine has enough history
        if len(unit_data) >= window_size:
            data_array = unit_data[feature_cols].values
            if target_col:
                target_array = unit_data[target_col].values
            
            for i in range(len(unit_data) - window_size + 1):
                sequences.append(data_array[i : i + window_size])
                if target_col:
                    targets.append(target_array[i + window_size - 1])
                    
    return np.array(sequences), np.array(targets) if target_col else np.array(sequences)