import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def get_model_params(dataset_id):
    """Returns optimized hyperparameters based on dataset complexity from your notebook."""
    params = {
        'n_estimators': 250,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Adaptive complexity for multi-regime vs single-regime
    if dataset_id in ['FD002', 'FD004']:
        params.update({'max_depth': 4, 'learning_rate': 0.03,'reg_alpha': 10,'reg_lambda':1, 'min_child_weight': 5})
    else:
        params.update({'max_depth': 6, 'learning_rate': 0.03, 'reg_alpha': 10,'reg_lambda':10,  'min_child_weight': 1})
        
    return params

def compute_nasa_score(y_true, y_pred):
    """Calculates the asymmetric NASA error score."""
    d = y_pred - y_true
    score = 0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13) - 1
        else:
            score += np.exp(diff / 10) - 1
    return score

def train_model(X_train, y_train, dataset_id):
    """Trains the XGBoost regressor using dataset-specific params."""
    params = get_model_params(dataset_id)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_on_test(model, test_df, y_truth, features):
    """Evaluates the model on the last cycle of each engine unit."""
    # Group by unit_id and take the last available cycle
    test_last = test_df.groupby('unit_id').last().reset_index()
    X_test = test_last[features]
    
    # Predict and clip at 0 to avoid negative RUL
    y_pred = np.maximum(model.predict(X_test), 0)
    
    rmse = np.sqrt(mean_squared_error(y_truth, y_pred))
    score = compute_nasa_score(y_truth, y_pred)
    
    return rmse, score, y_pred