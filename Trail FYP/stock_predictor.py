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
        os.makedirs(self.model_dir, exist_ok=True)
        
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
        self.model_initialized = False
        
    def create_model(self):
        """Create LSTM model for stock prediction"""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(50),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model_initialized = True
        
    def load_model(self, symbol: str):
        """Load a pre-trained model for a specific symbol"""
        try:
            self.model = self.model_manager.load_model(f"{symbol}_model.h5")
            self.model_initialized = True
        except:
            self.create_model()
            self.model_initialized = True
            
    def save_model(self, symbol: str):
        """Save the trained model"""
        if self.model_initialized:
            self.model_manager.save_model(self.model, f"{symbol}_model.h5")
            
    def load_data(self, symbol: str):
        """
        Load stock data from Alpha Vantage API
        """
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'apikey': self.api_key,
            'interval': '5min',
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
        df = df.sort_index()
        
        return df
        
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
        if not self.model_initialized:
            self.create_model()
            
        x, y = self.prepare_data(df)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        
    def predict(self, symbol: str, date: str = None):
        """
        Predict stock price for a given symbol
        
        Args:
            symbol: Stock symbol to predict
            date: Optional date for prediction (default: today)
            
        Returns:
            Dictionary containing prediction information
        """
        try:
            # Load data
            df = self.load_data(symbol)
            
            # Load or create model
            self.load_model(symbol)
            
            # Get the latest data point
            latest_data = df['Close'].values[-self.look_back:].reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(latest_data)
            
            # Prepare input for prediction
            x_input = scaled_data.reshape(1, self.look_back, 1)
            
            # Make prediction
            predicted_price_scaled = self.model.predict(x_input)
            predicted_price = self.scaler.inverse_transform(predicted_price_scaled)[0][0]
            
            # Get prediction date
            if date is None:
                prediction_date = (pd.to_datetime(df.index[-1]) + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                prediction_date = date
            
            return {
                "predicted_price": float(predicted_price),
                "prediction_date": prediction_date
            }
            
        except Exception as e:
            raise ValueError(f"Error predicting stock price: {str(e)}")

    def evaluate_model(self, df: pd.DataFrame):
        """
        Evaluate the performance of the LSTM model
        """
        if not self.model_initialized:
            self.create_model()
            
        x, y = self.prepare_data(df)
        loss = self.model.evaluate(x, y)
        return loss

    def save_model(self, filename: str):
        """
        Save the trained LSTM model
        """
        if self.model_initialized:
            self.model_manager.save_model(self.model, filename)

    def load_model(self, filename: str):
        """
        Load a trained LSTM model
        """
        self.model = self.model_manager.load_model(filename)
