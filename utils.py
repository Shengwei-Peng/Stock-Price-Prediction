import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple


def parse_args()-> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--window_size", default=20, type=int)
    args = parser.parse_args()
    
    return args


def sliding_window(data: pd.DataFrame, window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    
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
