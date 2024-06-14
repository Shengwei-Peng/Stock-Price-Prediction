import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_args()-> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=None, type=str, required=True)
    parser.add_argument("--test_path", default=None, type=str, required=True)
    parser.add_argument("--window_size", default=20, type=int)
    args = parser.parse_args()
    
    return args


def pre_process(path: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data = data.dropna()
    x = []
    y = []

    for i in tqdm(range(len(data) - window_size), desc="Processing data"):
        x.append(data[[
            'capacity',
            'turnover', 
            'open',
            'high',
            'low',
            'close',
            'change',
            'transaction',
            ]].iloc[i:i+window_size].values.flatten())
        y.append(data['close'].iloc[i+window_size])

    return np.array(x), np.array(y)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'R² (Coefficient of Determination)': r2,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae,
        'Mean Absolute Percentage Error (MAPE)': mape,
    }
    print("Model Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics
