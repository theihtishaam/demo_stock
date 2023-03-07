import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from flask import Flask, render_template

# Create Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    # Get data from Yahoo Finance
    df = yf.download('SPY', start='2022-01-01')

    # Show top 7 records
    print(df.head(7))

    # Set lookback period
    lookback = 9

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

    # Plot actual prices
    fig, ax = plt.subplots()

    ax.set_title('Close Price and Volume')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Get test set dates
    test_dates = df.index[train_size+lookback+1:]

    # Plot actual and predicted prices
    plt.plot(test_dates, actuals, label='Actual')
    plt.plot(test_dates[:len(predictions)], predictions, label='Predicted')
    plt.title('Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot volume
    ax2 = ax.twinx()
    ax2.plot(df.index, df['Volume'], color='r', label='Volume')
    ax2.set_ylabel('Volume')

    # Format x-axis ticks as dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%D-%Y'))

    # Save plot to image file
    plt.savefig('static/plot.png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
