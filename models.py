import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class linear_regression():
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = self.model.predict(x_test)
        return y_pred


class svm():
    def __init__(self):
        super().__init__()
        self.model = SVR()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = self.model.predict(x_test)
        return y_pred


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


class networks():
    def __init__(self, model: str, input_shape: Tuple[int, ...], batch_size: int = 4096, epochs: int = 1000, lr: float = 0.001):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model == "lstm":
            self.model = LSTMModel(input_size=input_shape[2]).to(self.device)
        if model == "transformer":
            self.model = TransformerModel(d_model=input_shape[2], nhead=input_shape[2]//2).to(self.device)

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


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 20, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.model_type = 'LSTM'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(x[:, -1, :])
        return x


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int = 2, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
