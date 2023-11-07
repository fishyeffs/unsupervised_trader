import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

def printOutputs(day, type, y_val, binary_predictions_val_1_day):
    accuracy = accuracy_score(y_val, binary_predictions_val_1_day)
    precision = precision_score(y_val, binary_predictions_val_1_day)
    recall = recall_score(y_val, binary_predictions_val_1_day)
    f1 = f1_score(y_val, binary_predictions_val_1_day)

    print(day + "-Day Predictions - " + type +" Set Metrics:")
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1 Score: {:.2f}".format(f1))

class StockPredictor:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(input_dim, 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, X_train, y_train, epochs):
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

    # Split into training, validation, and test datasets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

load = input("Load model? (y/n) ")
save = input("Save model? (y/n) ")
epochs = int(input("Number of epochs: "))

# Usage:
df = pd.read_csv(sys.path[0] + '/../BTC-2017min.csv')
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df)

# Create instances of the class for 1-day and 7-day predictions
predictor_1_day = StockPredictor(input_dim=6)
predictor_7_day = StockPredictor(input_dim=6)

# Load or train models for both horizons
if load == 'y':
    predictor_1_day.model = tf.keras.models.load_model("models/btc_model_1_day.keras")
    predictor_7_day.model = tf.keras.models.load_model("models/btc_model_7_day.keras")
else:
    try:
        # Train models for 1-day and 7-day predictions
        predictor_1_day.train(X_train, y_train, epochs)
        predictor_7_day.train(X_train, y_train, epochs)
    except ValueError as err:
        print('Train error', err)

# Validate both models on the validation set
val_likelihoods_1_day = predictor_1_day.predict(X_val)
val_likelihoods_7_day = predictor_7_day.predict(X_val)

# Make predictions for both horizons on the test set
test_likelihoods_1_day = predictor_1_day.predict(X_test)
test_likelihoods_7_day = predictor_7_day.predict(X_test)

if save == 'y':
    predictor_1_day.model.save("model/btc_model_1_day.keras")
    predictor_7_day.model.save("model/btc_model_7_day.keras")

# Convert predictions to binary (0 or 1) for both horizons
binary_predictions_val_1_day = (val_likelihoods_1_day > 0.5).astype(int)
binary_predictions_val_7_day = (val_likelihoods_7_day > 0.5).astype(int)

binary_predictions_test_1_day = (test_likelihoods_1_day > 0.5).astype(int)
binary_predictions_test_7_day = (test_likelihoods_7_day > 0.5).astype(int)

printOutputs("1", "Validation", y_val, binary_predictions_val_1_day)

printOutputs("1", "Test", y_val, binary_predictions_val_7_day)

printOutputs("7", "Validation", y_test, binary_predictions_test_1_day)

printOutputs("7", "Test", y_test, binary_predictions_test_7_day)
