import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name']
    lookback = int(request.form['lookback'])

    # Get data from Yahoo Finance
    df = yf.download(stock_name, start='2022-01-01')

    # Set lookback period
    lookback = lookback

    # Function to create dataset
    def create_dataset(dataset):
        X, Y = [], []
        for i in range(len(dataset)-lookback-1):
            X.append(dataset[i:(i+lookback), 0])
            Y.append(dataset[i+lookback, 0])
        return np.array(X), np.array(Y)

    # Scale data
    data = np.array(df['Close']).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Train and test split
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_X, train_Y = create_dataset(train_data)
    test_X, test_Y = create_dataset(test_data)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Create model
    model = Sequential()
    model.add(LSTM(50, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs=50, batch_size=64, verbose=2)

    # Make predictions on test data
    test_X = test_X.reshape((test_X.shape[0], lookback, 1))
    predictions = model.predict(test_X)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(test_Y.reshape(-1, 1))

    # Convert dates to datetime format
    df.index = pd.to_datetime(df.index)

    # Get test set dates
    test_dates = df.index[train_size+lookback+1:]

    # Create list of predicted prices and dates
    prediction_dates = []
    prediction_prices = []
    for i in range(len(predictions)):
        prediction_dates.append(test_dates[i])
        prediction_prices.append(round(predictions[i][0], 2))

    return render_template('predict.html', stock_name=stock_name, lookback=lookback, prediction_dates=prediction_dates, prediction_prices=prediction_prices)

if __name__ == '__main__':
    app.run(debug=True)
