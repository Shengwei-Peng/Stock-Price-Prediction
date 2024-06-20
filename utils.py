import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from argparse import Namespace, ArgumentParser
from models import linear_regression, svm, random_forest, xgboost, networks

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args()-> Namespace:

    parser = ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--model", default="catboost", type=str)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--test_size", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    
    return args

class Stocker():
    def __init__(self, args: Namespace):
        self.args = args
        set_seed(self.args.seed)
    def preprocess(self):
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
                'change',
                'transaction',
                'close',
                ]].iloc[i : i + self.args.window_size].values)
            y.append(data['change'].iloc[i + self.args.window_size] / data['close'].iloc[i + self.args.window_size -1])
        x = np.array(x)
        y = np.array(y)
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=self.args.test_size, shuffle=False)
        y_test = data['close'].iloc[-self.args.test_size:]
        
        print(f"x_train: {x_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_test: {y_test.shape}")

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        if self.args.model == "linear_regression":
            self.model = linear_regression()
        elif self.args.model == "svm":
            self.model = svm()
        elif self.args.model == "random_forest":
            self.model = random_forest(self.args.seed)
        elif self.args.model == "xgboost":
            self.model = xgboost(self.args.seed)
        else:
            self.model = networks(model=self.args.model, input_shape=self.x_train.shape)
        
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = (1 + self.y_pred) * self.x_test[:,-1,-1]

    def evaluate(self):
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        ndei = rmse / np.std(self.y_test)
        self.metrics = {
            'R² (Coefficient of Determination)': r2,
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'Mean Absolute Error (MAE)': mae,
            'Non-dimensional Error Index (NDEI)': ndei,
        }
        print("Model Performance Metrics:")
        for key, value in self.metrics.items():
            print(f"{key:<35}: {value:>10.4f}")

    def visualize(self):
        date = self.data['date'].iloc[-len(self.y_pred):]
        plt.figure(figsize=(14, 7))
        plt.plot(date, self.y_test, label='True Price', color='blue', linestyle='-', linewidth=2)
        plt.plot(date, self.y_pred, label='Predicted Price', color='red', linestyle='--', linewidth=2)
        plt.title('Stock Price Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Stock Price', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(14, 7))
        plt.plot(date, residuals, label='Residuals', color='purple', linestyle='-', linewidth=2)
        plt.title('Residuals over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Residuals', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plt.figure(figsize=(14, 7))
        plt.scatter(self.y_test, self.y_pred, label='Predicted vs True', color='green')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--', linewidth=2)
        plt.title('Predicted vs True Prices', fontsize=16, fontweight='bold')
        plt.xlabel('True Prices', fontsize=14)
        plt.ylabel('Predicted Prices', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
