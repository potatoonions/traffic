import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import tensorflow as tf

load_dotenv()

class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def save_model(self, model, filename):
        model.save(os.path.join(self.model_dir, filename))

    def load_model(self, filename):
        return tf.keras.models.load_model(os.path.join(self.model_dir, filename))

class StockPredictor:
    def __init__(self, model_dir):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 60  # Number of previous days to consider
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.model_manager = ModelManager(model_dir)
        
    def load_data(self, symbol: str):
        """
        Load stock data from Alpha Vantage API
        """
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'apikey': self.api_key,
            'interval': '5min',  # 5-minute intervals for real-time data
            'outputsize': 'full'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Time Series (5min)' not in data:
            raise ValueError(f"No data found for symbol {symbol}")
            
        df = pd.DataFrame(data['Time Series (5min)']).T
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        return df
    
    def create_model(self):
        """
        Create LSTM model for stock prediction
        """
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare data for LSTM model
        """
        close_prices = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        
        x, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        return x, y
    
    def train_model(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the LSTM model
        """
        x, y = self.prepare_data(df)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def predict(self, symbol: str, days_to_predict: int = 1):
        """
        Make stock price predictions
        """
        df = self.load_data(symbol)
        
        # Get the last 60 days of data
        last_60_days = df['Close'].tail(self.look_back).values.reshape(-1, 1)
        last_60_days_scaled = self.scaler.transform(last_60_days)
        
        # Create input for prediction
        x_test = []
        x_test.append(last_60_days_scaled)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Make prediction
        predicted_price = self.model.predict(x_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        
        return {
            'symbol': symbol,
            'predicted_price': float(predicted_price[0][0]),
            'current_price': float(df['Close'].iloc[-1]),
            'prediction_date': (datetime.now() + timedelta(days=days_to_predict)).strftime('%Y-%m-%d')
        }

    def evaluate_model(self, df: pd.DataFrame):
        """
        Evaluate the performance of the LSTM model
        """
        x, y = self.prepare_data(df)
        loss = self.model.evaluate(x, y)
        return loss

    def save_model(self, filename: str):
        """
        Save the trained LSTM model
        """
        self.model_manager.save_model(self.model, filename)

    def load_model(self, filename: str):
        """
        Load a trained LSTM model
        """
        self.model = self.model_manager.load_model(filename)
