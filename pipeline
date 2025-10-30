import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
import os
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define constants
TICKER = "NVDA"          # Stock symbol
LOOKBACK = 30            # Days to look back
EPOCHS = 50              # Training epochs
BATCH_SIZE = 16          # Batch size
LEARNING_RATE = 0.001    # Learning rate
TRAIN_SPLIT = 0.8        # Train/test split
N_QUBITS = 4             # Number of qubits
N_LAYERS = 2             # Number of quantum layers
HIDDEN_SIZE = 64         # LSTM hidden size
OUTPUT_DIR = "outputs"   # Output directory

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set up plotting style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Define Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Alternative QLayer implementation that avoids TorchLayer initialization issues
class QLayer(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize weights directly as parameters
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
    def forward(self, x):
        batch_size = x.shape[0]
        processed_data = torch.zeros(batch_size, self.n_qubits, device=x.device)
        
        # Define the quantum circuit function
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Encode inputs
            for j in range(self.n_qubits):
                qml.RY(inputs[j], wires=j)
            
            # Variational layers
            for l in range(self.n_layers):
                # Entangling layer
                for j in range(self.n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
                
                # Rotation layer
                for j in range(self.n_qubits):
                    qml.RX(weights[l, j, 0], wires=j)
                    qml.RY(weights[l, j, 1], wires=j)
                    qml.RZ(weights[l, j, 2], wires=j)
            
            # Measurements
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]
        
        # Process each batch element
        for i in range(batch_size):
            # Prepare inputs
            if x.shape[1] >= self.n_qubits:
                inputs = x[i, :self.n_qubits].detach().cpu().numpy()
            else:
                inputs = torch.zeros(self.n_qubits)
                inputs[:x.shape[1]] = x[i]
                inputs = inputs.detach().cpu().numpy()
            
            # Run circuit
            weights_np = self.weights.detach().cpu().numpy()
            result = circuit(inputs, weights_np)
            
            # Store results
            processed_data[i] = torch.tensor(result, device=x.device)
        
        return processed_data

# Define Quantum LSTM Cell
class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Quantum layers for gates - FIXED: added parentheses
        self.forget_quantum = QLayer()
        self.input_quantum = QLayer()
        self.output_quantum = QLayer()
        self.cell_quantum = QLayer()
        
        # Linear layers for quantum output processing
        self.forget_linear = nn.Linear(N_QUBITS, hidden_size)
        self.input_linear = nn.Linear(N_QUBITS, hidden_size)
        self.output_linear = nn.Linear(N_QUBITS, hidden_size)
        self.cell_linear = nn.Linear(N_QUBITS, hidden_size)
        
        # Projections to quantum dimensions
        self.forget_projection = nn.Linear(input_size + hidden_size, N_QUBITS)
        self.input_projection = nn.Linear(input_size + hidden_size, N_QUBITS)
        self.output_projection = nn.Linear(input_size + hidden_size, N_QUBITS)
        self.cell_projection = nn.Linear(input_size + hidden_size, N_QUBITS)
    
    def forward(self, x, hidden):
        # Unpack hidden states
        h_prev, c_prev = hidden
        
        # Combine input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Project to quantum dimensions
        forget_input = self.forget_projection(combined)
        input_input = self.input_projection(combined)
        output_input = self.output_projection(combined)
        cell_input = self.cell_projection(combined)
        
        # Apply quantum layers
        forget_quantum = self.forget_quantum(forget_input)
        input_quantum = self.input_quantum(input_input)
        output_quantum = self.output_quantum(output_input)
        cell_quantum = self.cell_quantum(cell_input)
        
        # Apply linear transformations and activations
        forget_gate = torch.sigmoid(self.forget_linear(forget_quantum))
        input_gate = torch.sigmoid(self.input_linear(input_quantum))
        output_gate = torch.sigmoid(self.output_linear(output_quantum))
        cell_gate = torch.tanh(self.cell_linear(cell_quantum))
        
        # LSTM equations
        c_next = forget_gate * c_prev + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next

# Define full QLSTM model
class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(QLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # QLSTM cells
        self.cells = nn.ModuleList([
            QLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Get dimensions
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for i in range(self.num_layers):
                if i > 0:
                    x_t = h[i-1]
                
                h[i], c[i] = self.cells[i](x_t, (h[i], c[i]))
        
        # Final prediction
        out = self.fc(h[-1])
        return out

# Function to fetch stock data
def fetch_stock_data(ticker, period="5y"):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    print(f"Retrieved {len(data)} data points")
    return data

# Function to prepare data
def prepare_data(data, lookback=30, train_split=0.8):
    # Select features
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - lookback):
        X.append(features_scaled[i:i+lookback])
        y.append(features_scaled[i+lookback, 3])  # Close price
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split data
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    return X_train, y_train, X_test, y_test, scaler

# Function to train model
def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=50):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Added progress bar
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return train_losses, test_losses

# Function to predict future prices
def predict_future(model, last_sequence, scaler, prediction_days=30):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone().detach()
    
    # Move to CPU for numpy operations
    if current_sequence.device != torch.device('cpu'):
        current_sequence = current_sequence.cpu()
    
    # Make predictions
    for _ in range(prediction_days):
        with torch.no_grad():
            # Prepare input
            current_sequence_gpu = current_sequence.unsqueeze(0).to(device)
            # Get prediction
            pred = model(current_sequence_gpu)
            pred = pred.cpu().item()
            
            # Store prediction
            predictions.append(pred)
            
            # Update sequence for next prediction
            new_sequence = current_sequence.clone()
            new_sequence[:-1] = current_sequence[1:]
            new_sequence[-1, 3] = pred  # Update close price
            
            current_sequence = new_sequence
    
    # Convert to actual prices
    pred_array = np.zeros((len(predictions), 5))
    pred_array[:, 3] = np.array(predictions)
    
    # Inverse transform
    inverse_predictions = scaler.inverse_transform(pred_array)[:, 3]
    
    return inverse_predictions

# Function to calculate future dates
def calculate_future_dates(start_date, num_days):
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    business_dates = [date for date in dates if date.weekday() < 5]
    return business_dates[:num_days]

# Main function
def main():
    # Fetch data
    data = fetch_stock_data(TICKER)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(data, lookback=LOOKBACK, train_split=TRAIN_SPLIT)
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = QLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=1, output_size=1)
    model = model.to(device)
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Train model
    print("Training model...")
    train_losses, test_losses = train_model(model, train_loader, test_loader, optimizer, criterion, epochs=EPOCHS)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
    plt.close()
    
    # Evaluate on test data
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.cpu().numpy().flatten())
    
    # Convert to original scale
    pred_array = np.zeros((len(predictions), 5))
    actual_array = np.zeros((len(actuals), 5))
    pred_array[:, 3] = np.array(predictions)
    actual_array[:, 3] = np.array(actuals)
    
    pred_prices = scaler.inverse_transform(pred_array)[:, 3]
    actual_prices = scaler.inverse_transform(actual_array)[:, 3]
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual Prices', alpha=0.7)
    plt.plot(pred_prices, label='Predicted Prices', alpha=0.7)
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.title(f'{TICKER} Stock Price Prediction')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_vs_actual.png'))
    plt.close()
    
    # Predict future prices
    last_sequence = X_test[-1]
    
    # Different time frames for prediction
    future_days = {
        'day': 30,       # 1 month daily
        'week': 7 * 4,   # 4 weeks
        'month': 30,     # 1 month
        '3month': 90,    # 3 months
        '6month': 180,   # 6 months
        'year': 365      # 1 year
    }
    
    # Current date
    today = datetime.now()
    
    # Make predictions for each time frame
    print("\nFuture price predictions:")
    for period, days in future_days.items():
        predictions = predict_future(model, last_sequence, scaler, days)
        
        # Calculate dates
        future_dates = calculate_future_dates(today, days)
        
        # Plot predictions
        plt.figure(figsize=(14, 7))
        plt.plot(future_dates, predictions, label=f'Predicted Prices ({period})*- coding: utf-8 -*-', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title(f'{TICKER} Stock Price Prediction ({period})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_{period}.png'))
        plt.close()
        
        # Print predictions
        last_day_pred = predictions[-1]
        print(f"{period.capitalize()} prediction (on {future_dates[-1].strftime('%Y-%m-%d')}): ${last_day_pred:.2f}")
    
    print("\nAnalysis complete! Check the 'outputs' folder for visualizations.")

# Run the main function
if __name__ == "__main__":
    main()
