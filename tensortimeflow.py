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
    print(df.dtypes)
    df['unix'] = df['unix'].astype(int) // 10**9

    # Sort by date
    df = df.sort_values('unix')

    # Drop date column
    df = df.drop('symbol', axis=1)
    df = df.drop('date', axis=1)

    # Define features and labels
    X = df.drop('close', axis=1)
    y = df['close']

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape features for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Convert labels to binary
    y = np.where(y.pct_change() > 0, 1, 0).astype(float)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

load = input("Load model? (y/n) ")
save = input("Save model? (y/n) ")
epochs = int(input("Number of epochs: "))

# Usage:
df = pd.read_csv('BTC-2017min.csv')
X_train, y_train, X_test, y_test = preprocess_data(df)

# Create an instance of the class
predictor = StockPredictor(input_dim=6)
if load is 'y':
    predictor.model = tf.keras.models.load_model("btc_model.keras")
else:
    try:
        predictor.train(X_train, y_train, epochs)
    except ValueError as err:
        print('Train error', err)

# Train the model

# Make predictions
likelihoods = predictor.predict(X_test)
if save is 'y':
    predictor.model.save("btc_model.keras")

print(likelihoods)
print(likelihoods.size)
meanVal = np.mean(likelihoods)
print("Mean value: {:.2f}".format(meanVal))
