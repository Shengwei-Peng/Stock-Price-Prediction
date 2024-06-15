import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def parse_args()-> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--test_size", default=20, type=int)
    args = parser.parse_args()
    
    return args

def pre_process(
        path: str, 
        window_size: int,
        test_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
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
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")

    return x_train, x_test, y_train, y_test

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
        print(f"{key:<40}: {value:>10.4f}")

    return metrics
