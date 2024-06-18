import torch
import numpy as np
import torch.nn as nn
from catboost import CatBoostRegressor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class random_forest():
    def __init__(self, seed):
        super().__init__()
        self.model = RandomForestRegressor(random_state=seed)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = self.model.predict(x_test)
        return y_pred


class xgboost():
    def __init__(self, seed):
        super().__init__()
        self.model = XGBRegressor(random_state=seed)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = self.model.predict(x_test)
        return y_pred


class catboost():
    def __init__(self, seed):
        super().__init__()
        self.model = CatBoostRegressor(random_seed=seed)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = self.model.predict(x_test)
        return y_pred


class networks():
    def __init__(self, model: str, input_shape: Tuple[int, ...], batch_size: int = 4096, epochs: int = 1000, lr: float = 0.001):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model == "lstm":
            self.model = lstm(input_size=input_shape[2], hidden_size=20, num_layers=2).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train_torch = torch.tensor(x_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        train_data = TensorDataset(x_train_torch, y_train_torch)
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test_torch = torch.tensor(x_test, dtype=torch.float32)
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for inputs in x_test_torch:
                inputs = inputs.unsqueeze(0).to(self.device)
                outputs = self.model(inputs)
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_pred = np.squeeze(y_pred)
        return y_pred


class lstm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(x[:, -1, :])
        return x

class mlp(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        return x