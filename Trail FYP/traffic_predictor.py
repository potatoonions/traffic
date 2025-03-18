import numpy as np
import tensorflow as tf
from typing import List, Tuple

class TrafficPredictor:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = None  # Will be used for data normalization
        
    def _build_model(self) -> tf.keras.Model:
        """Build and return an LSTM model for traffic prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(24, 4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the model on historical traffic data"""
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        return history
    
    def predict_traffic(self, location_data: np.ndarray) -> float:
        """Predict traffic intensity for a given location and time"""
        if location_data.ndim == 2:
            location_data = location_data[np.newaxis, :]
            
        prediction = self.model.predict(location_data)
        return float(prediction[0, 0])
    
    def prepare_data(self, raw_data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare raw traffic data for training
        
        Args:
            raw_data: List of tuples containing (timestamp, latitude, longitude, speed, volume)
        
        Returns:
            Tuple of (X, y) where X is the input features and y is the target variable
        """
        data = np.array(raw_data)
        
        # Create sequences of 24 time steps
        X, y = [], []
        for i in range(len(data) - 24):
            X.append(data[i:i+24, :4])  # time, lat, lon, speed
            y.append(data[i+24, 3])     # next speed value
            
        return np.array(X), np.array(y)
    
    def save_model(self, path: str):
        """Save the trained model to disk"""
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load a trained model from disk"""
        self.model = tf.keras.models.load_model(path)
