import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define dataset path
dataset_path = r"C:\Users\DELL\PycharmProjects\Deep Learning\Time series Dataset"
train_file = f"{dataset_path}\DailyDelhiClimateTrain.csv"
test_file = f"{dataset_path}\DailyDelhiClimateTest.csv"

# Load datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Convert date column to datetime and sort
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

train_df = train_df.sort_values(by='date')
test_df = test_df.sort_values(by='date')

# Select features and target variable
features = ['humidity', 'wind_speed', 'meanpressure']
target = 'meantemp'

# Normalize data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

train_features = scaler_x.fit_transform(train_df[features])
train_target = scaler_y.fit_transform(train_df[[target]])

test_features = scaler_x.transform(test_df[features])
test_target = scaler_y.transform(test_df[[target]])


# Function to create sequences
def create_sequences(data_x, data_y, seq_length):
    sequences = []
    for i in range(len(data_x) - seq_length):
        seq_x = data_x[i:i + seq_length]
        seq_y = data_y[i + seq_length]
        sequences.append((seq_x, seq_y))
    return sequences


# Sequence length
seq_length = 10

# Create sequences for training and testing
train_sequences = create_sequences(train_features, train_target, seq_length)
test_sequences = create_sequences(test_features, test_target, seq_length)


# Define PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_x, seq_y = self.sequences[idx]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


# Create DataLoaders
train_dataset = TimeSeriesDataset(train_sequences)
test_dataset = TimeSeriesDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use output of last time step
        return out


# Model Parameters
input_size = len(features)  # Number of input features
hidden_size = 64
num_layers = 2
learning_rate = 0.001
num_epochs = 20

# Initialize Model
model = RNNModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for seq_x, seq_y in train_loader:
        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, seq_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate Model
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for seq_x, seq_y in test_loader:
        output = model(seq_x).numpy()
        predictions.append(output[0, 0])
        actuals.append(seq_y.numpy()[0, 0])

# Inverse transform predictions and actuals
predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# Compute Evaluation Metrics
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)

print("\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
