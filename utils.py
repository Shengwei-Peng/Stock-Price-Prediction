import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple


def sliding_window(data: pd.DataFrame, window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
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
