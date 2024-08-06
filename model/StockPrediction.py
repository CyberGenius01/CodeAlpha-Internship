import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import json

with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'r') as f:
    jdata = json.load(f)
files = jdata[0]['files']
evals = jdata[1]['evals']

# Step 1: Load the Data
# Replace 'AAPL' with the stock ticker you want to predict
df = yf.download('AAPL', start='2012-01-01', end='2024-01-01')

# Step 2: Preprocess the Data
# We'll use the 'Close' column for prediction
data = df['Close'].values.reshape(-1, 1)

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define a function to create sequences
def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Sequence length to look back
sequence_length = 60

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for training and testing
x_train, y_train = create_sequences(train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)

# Reshape data to be 3D for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 3: Build the LSTM Model
model = tf.keras.models.Sequential()

# Add first LSTM layer
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))

# Add second LSTM layer
model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))

# Add a Dense layer
model.add(tf.keras.layers.Dense(units=25))
model.add(tf.keras.layers.Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
model.fit(x_train, y_train, batch_size=32, epochs=20)

# Step 5: Make Predictions
# Get the model's predicted prices on the test data
predictions = model.predict(x_test)

# Unscale the data (inverse transform)
predictions = scaler.inverse_transform(predictions)

# Unscale the true test data for comparison
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Visualize the Results
plt.style.use('dark_background')
plt.figure(figsize=(14, 5))
plt.plot(y_test_unscaled, color='cyan', label='Actual Stock Price')
plt.plot(predictions, color='crimson', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig(files['stock'], transparent=True)

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_unscaled, predictions)
evals['stock_mse'] = f'{(mse*10):.2f}%';

with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'w') as f:
    json.dump(jdata, f, indent=4)