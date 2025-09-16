import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess data
df = pd.read_csv('gold_prices.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# --- training: multi-output targets (High, Low, Open, Close) ---
FEATURES = ['High', 'Low', 'Open', 'Close']
OUTPUT_FEATURES = ['High', 'Low', 'Open', 'Close']  # same targets

data = df[FEATURES].values
print(data)
# scaler as before
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


def create_sequences_multi(data, seq_length, output_idx):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, :])  # X: seq_length x n_features
        y.append(data[i, output_idx])  # y: vector of target columns
    return np.array(X), np.array(y)


# indices of OUTPUT_FEATURES in FEATURES
output_idx = [FEATURES.index(c) for c in OUTPUT_FEATURES]  # [0,1,2,3]

SEQ_LENGTH = 60
X, y = create_sequences_multi(scaled_data, SEQ_LENGTH, output_idx)

# Train-test split same as before
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model with multi-output
n_targets = y_train.shape[1]  # should be 4
model = Sequential([
    LSTM(500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(250, return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(n_targets)
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=13,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# Create inverse transformation function for multi-output
def inverse_transform_multi_predictions(predictions, scaler, data, output_idx):
    """
    Inverse transform multi-output predictions
    predictions: array of shape (n_samples, n_targets)
    returns: array of shape (n_samples, n_targets) with inverse scaled values
    """
    n_samples = predictions.shape[0]
    n_features = data.shape[1]

    # Create dummy array with same shape as original data
    dummy_array = np.zeros((n_samples, n_features))

    # Place predictions in their respective columns
    for i, idx in enumerate(output_idx):
        dummy_array[:, idx] = predictions[:, i]

    # Inverse transform
    inverse_scaled = scaler.inverse_transform(dummy_array)

    # Extract only the target columns
    return inverse_scaled[:, output_idx]


# Inverse scale actual and predicted values
y_train_inv = inverse_transform_multi_predictions(y_train, scaler, data, output_idx)
y_test_inv = inverse_transform_multi_predictions(y_test, scaler, data, output_idx)
train_predict_inv = inverse_transform_multi_predictions(train_predict, scaler, data, output_idx)
test_predict_inv = inverse_transform_multi_predictions(test_predict, scaler, data, output_idx)

# Get dates for plotting
train_dates = df.index[SEQ_LENGTH:SEQ_LENGTH + len(y_train)]
test_dates = df.index[SEQ_LENGTH + len(y_train):SEQ_LENGTH + len(y_train) + len(y_test)]

feature_names = ['High', 'Low', 'Open', 'Close']

# --- Plot results for entire dataset (train + test) ---
all_dates = df.index[SEQ_LENGTH:]  # because first SEQ_LENGTH days have no labels
all_actual = np.concatenate([y_train_inv, y_test_inv], axis=0)
all_predicted = np.concatenate([train_predict_inv, test_predict_inv], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    ax = axes[i]

    # Plot actual values
    ax.plot(all_dates, all_actual[:, i], label='Actual', color='blue', linewidth=2)

    # Plot predicted values
    ax.plot(all_dates, all_predicted[:, i], label='Predicted', color='red', linewidth=2, alpha=0.8)

    ax.set_title(f'{feature} Price - Actual vs Predicted (Full Dataset)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Plot candlestick-like comparison for a specific period
plt.figure(figsize=(16, 8))

# Select a subset of test data for clearer visualization (last 100 days)
subset_size = min(100, len(test_dates))
subset_dates = test_dates[61:61+subset_size]
subset_actual = y_test_inv[61:61+subset_size]
subset_predicted = test_predict_inv[61:61+subset_size]

# Plot actual prices
plt.plot(subset_dates, subset_actual[:, 0], label='Actual High', color='green', linewidth=2)
plt.plot(subset_dates, subset_actual[:, 1], label='Actual Low', color='red', linewidth=2)
plt.plot(subset_dates, subset_actual[:, 2], label='Actual Open', color='blue', linewidth=2)
plt.plot(subset_dates, subset_actual[:, 3], label='Actual Close', color='orange', linewidth=2)

# Plot predicted prices with dashed lines
plt.plot(subset_dates, subset_predicted[:, 0], label='Predicted High', color='green', linestyle='--', linewidth=2)
plt.plot(subset_dates, subset_predicted[:, 1], label='Predicted Low', color='red', linestyle='--', linewidth=2)
plt.plot(subset_dates, subset_predicted[:, 2], label='Predicted Open', color='blue', linestyle='--', linewidth=2)
plt.plot(subset_dates, subset_predicted[:, 3], label='Predicted Close', color='orange', linestyle='--', linewidth=2)

plt.title('Gold Price Prediction - All Features (First 100 Days of Test Set)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('gold_price_lstm_multi_output_model.h5')
