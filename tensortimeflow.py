import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class StockPredictor:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(input_dim, 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

def preprocess_data(df):
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date')

    # Drop date column
    df = df.drop('Date', axis=1)

    # Define features and labels
    X = df.drop('Adj Close', axis=1)
    y = df['Adj Close']

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape features for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Convert labels to binary
    y = np.where(y.pct_change() > 0, 1, 0)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

# Usage:
df = pd.read_csv('alltimenasdaq.csv')
X_train, y_train, X_test, y_test = preprocess_data(df)

# Create an instance of the class
predictor = StockPredictor(input_dim=5)

# Train the model
try:
    predictor.train(X_train, y_train, epochs=100)
except ValueError as err:
    print('Train error', err)

# Make predictions
likelihoods = predictor.predict(X_test)

print(likelihoods)