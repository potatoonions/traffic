import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class StockTrendPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 60
        
    def create_model(self):
        """Create an LSTM model for trend prediction"""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for trend prediction"""
        close_prices = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        
        x, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        return x, y
        
    def train_model(self, df: pd.DataFrame, epochs: int = 50):
        """Train the trend prediction model"""
        x, y = self.prepare_data(df)
        self.model.fit(x, y, epochs=epochs, batch_size=32)
        
    def predict_trend(self, df: pd.DataFrame):
        """
        Predict stock trend (rise or fall)
        
        Returns:
            Dictionary containing trend prediction and confidence
        """
        # Get the latest 60 data points
        latest_data = df['Close'].values[-self.look_back:].reshape(-1, 1)
        scaled_data = self.scaler.transform(latest_data)
        
        # Prepare input for prediction
        x_input = scaled_data.reshape(1, self.look_back, 1)
        
        # Make prediction
        predicted_price_scaled = self.model.predict(x_input)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)[0][0]
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        
        # Calculate trend and confidence
        trend = "rise" if predicted_price > current_price else "fall"
        confidence = abs((predicted_price - current_price) / current_price) * 100
        
        return {
            "trend": trend,
            "confidence": float(confidence),
            "predicted_price": float(predicted_price),
            "current_price": float(current_price)
        }
