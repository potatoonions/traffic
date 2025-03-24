import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class StockTrainer:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 60  # Number of previous days to consider
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
    def load_data(self, symbol: str, interval: str = '1D'):
        """
        Load stock data from Alpha Vantage API
        """
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"No data found for symbol {symbol}")
            
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume'
        })
        
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
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
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train_model(self, symbol: str, epochs: int = 50):
        """
        Train the LSTM model for a specific stock
        """
        try:
            # Load data
            df = self.load_data(symbol)
            
            # Create and compile model
            self.create_model()
            
            # Prepare data
            X, y = self.prepare_data(df)
            
            # Train model
            self.model.fit(X, y, epochs=epochs, batch_size=32)
            
            # Save model
            self.model.save(f'models/{symbol}_model.h5')
            
            return {
                'status': 'success',
                'symbol': symbol,
                'message': f'Model trained successfully for {symbol}',
                'epochs': epochs
            }
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
