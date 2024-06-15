import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

def parse_args()-> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--test_size", default=20, type=int)
    args = parser.parse_args()
    
    return args

class Stocker():
    def __init__(self, args):
        self.args = args

    def pre_process(self):
        data = pd.read_csv(self.args.data_path)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        self.data = data.dropna()
        x = []
        y = []
        for i in tqdm(range(len(data) - self.args.window_size), desc="Processing data"):
            x.append(data[[
                'capacity',
                'turnover', 
                'open',
                'high',
                'low',
                'close',
                'change',
                'transaction',
                ]].iloc[i : i + self.args.window_size].values.flatten())
            y.append(data['change'].iloc[i + self.args.window_size])
        x = np.array(x)
        y = np.array(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.args.test_size, shuffle=False)
        print(f"x_train: {x_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_test: {y_test.shape}")

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        self.model = CatBoostRegressor(verbose=100)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

    def test(self):
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        self.metrics = {
            'R² (Coefficient of Determination)': r2,
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'Mean Absolute Error (MAE)': mae,
        }
        print("Model Performance Metrics:")
        for key, value in self.metrics.items():
            print(f"{key:<35}: {value:>10.4f}")

    def plot(self):
        date = self.data['date'].iloc[-len(self.y_pred):]
        plt.figure(figsize=(14, 7))
        plt.plot(date, self.y_test, label='True Price', color='blue', linestyle='-', linewidth=2)
        plt.plot(date, self.y_pred, label='Predicted Price', color='red', linestyle='--', linewidth=2)
        plt.title('TSMC Stock Price Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Stock Price', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
