import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class NASALoss(nn.Module):
    """
    Differentiable PyTorch implementation of the NASA asymmetric scoring function.
    Can be used directly with loss.backward().
    """
    def __init__(self):
        super(NASALoss, self).__init__()

    def forward(self, y_pred, y_true):
        d = y_pred.view(-1) - y_true.view(-1)
        # Asymmetric penalty: exp(-d/13)-1 for early, exp(d/10)-1 for late
        loss = torch.where(
            d < 0,
            torch.exp(-d / 13.0) - 1.0,
            torch.exp(d / 10.0) - 1.0
        )
        return torch.mean(loss)

class RUL_LSTM(nn.Module):
    """
    Many-to-One LSTM architecture for sequence-based RUL estimation.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(RUL_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        out, (hn, cn) = self.lstm(x)
        # We only care about the hidden state of the final time step
        return self.regressor(out[:, -1, :])

def prepare_lstm_data(df, window_size, feature_cols, target_col):
    """
    Transforms flat DataFrames into 3D sequences [samples, time_steps, features].
    """
    X, y = [], []
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        if len(unit_data) >= window_size:
            data = unit_data[feature_cols].values
            target = unit_data[target_col].values
            # Sliding window approach
            for i in range(len(unit_data) - window_size + 1):
                X.append(data[i:i+window_size])
                y.append(target[i+window_size-1])
                
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



def train_model_dl(X_train, y_train, model, batch_size=64, epochs=50, lr=0.001):
    """
    Standard PyTorch training loop using the custom NASALoss.
    """
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Using NASALoss to optimize specifically for the NASA benchmark
    criterion = NASALoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs} | Avg NASALoss: {epoch_loss/len(loader):.4f}")
            
    return model

def evaluate_lstm(model, test_df, y_truth, feature_cols, window_size):
    """
    Evaluates the LSTM on the terminal state of each engine in the test set.
    """
    model.eval()
    y_preds = []
    
    # C-MAPSS Test evaluation requires predicting on the last available window
    for unit_id in test_df['unit_id'].unique():
        unit_data = test_df[test_df['unit_id'] == unit_id]
        last_window = unit_data[feature_cols].values[-window_size:]
        
        # Reshape for model input: [1, seq_len, features]
        x_input = torch.from_numpy(last_window).float().unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x_input).item()
            y_preds.append(max(0, pred))
            
    y_preds = np.array(y_preds)
    rmse = np.sqrt(((y_preds - y_truth) ** 2).mean())
    
    # Calculate final NASA Score for summary
    diff = y_preds - y_truth
    nasa_score = np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    
    return rmse, nasa_score