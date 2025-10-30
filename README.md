# Quantum-Enhanced LSTM for Stock Price Prediction

This project implements a quantum-enhanced Long Short-Term Memory (QLSTM) neural network for predicting NVIDIA stock prices across various timeframes.

## Overview

Traditional LSTM networks are powerful for time series prediction, but they can be enhanced with quantum computing techniques. This model implements a hybrid quantum-classical approach where the gates within the LSTM cells are augmented using quantum circuits implemented with PennyLane.

![Prediction Example](outputs/prediction_vs_actual.png)

## Features

- **Quantum Enhancement**: Uses variational quantum circuits to enhance LSTM gate operations
- **Multiple Timeframes**: Predicts stock prices for day, week, month, 3-month, 6-month, and yearly horizons
- **GPU Acceleration**: Automatically uses GPU if available
- **Visualization**: Generates plots for training metrics and predictions
- **Progress Tracking**: Includes training progress bars using tqdm

## Requirements

- Python 3.7+
- PyTorch
- PennyLane
- yfinance
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- tqdm

Install required packages:

```bash
pip install torch pennylane yfinance scikit-learn matplotlib seaborn pandas numpy tqdm
```

## Usage

Simply run the main script:

```bash
pipeline.py
```

The script will:
1. Download NVIDIA historical stock data
2. Prepare and normalize the data
3. Train the quantum-enhanced LSTM model
4. Evaluate performance on test data
5. Generate future price predictions
6. Save visualizations to the "outputs" directory

## Model Architecture

### Quantum Layer
- Uses quantum circuits with RY gates for data encoding
- Applies variational quantum circuits with entangling layers
- Measures qubit expectation values to extract quantum-processed features

### Quantum LSTM Cell
- Enhances all four LSTM gates (forget, input, output, and cell) with quantum layers
- Projects classical features to quantum dimensions
- Processes quantum features through linear layers

### Full QLSTM Model
- Processes input sequences through quantum-enhanced LSTM cells
- Outputs price predictions for various timeframes

## Customization

The model has several parameters that can be adjusted:
- `TICKER`: Stock symbol (default: "NVDA" for NVIDIA)
- `LOOKBACK`: Number of days to use for prediction (default: 30)
- `EPOCHS`: Number of training iterations (default: 50)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `N_QUBITS`: Number of qubits in quantum circuit (default: 4)
- `N_LAYERS`: Number of variational layers (default: 2)
- `HIDDEN_SIZE`: LSTM hidden layer size (default: 64)

## Outputs

The script generates several visualizations in the "outputs" directory:
- Training and validation loss curves
- Test data predictions vs actual prices
- Future price predictions for various timeframes

## Acknowledgements

This implementation is inspired by research on quantum-enhanced recurrent neural networks, particularly quantum LSTM architectures.

## Disclaimer

This project is for educational purposes only. The predictions should not be used for actual trading decisions, as stock markets are influenced by many factors beyond historical price patterns.
