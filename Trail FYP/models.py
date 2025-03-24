import os
from pathlib import Path
import tensorflow as tf
from typing import Dict, Any

class ModelManager:
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def save_model(self, model: tf.keras.Model, symbol: str) -> str:
        """
        Save a trained model to disk
        
        Args:
            model: The TensorFlow/Keras model to save
            symbol: Stock symbol associated with the model
            
        Returns:
            Path to the saved model file
        """
        model_path = self.models_dir / f'{symbol}_model.h5'
        model.save(model_path)
        return str(model_path)

    def load_model(self, symbol: str) -> tf.keras.Model:
        """
        Load a saved model from disk
        
        Args:
            symbol: Stock symbol associated with the model
            
        Returns:
            Loaded TensorFlow/Keras model
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        model_path = self.models_dir / f'{symbol}_model.h5'
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for symbol {symbol}")
        
        return tf.keras.models.load_model(model_path)

    def list_models(self) -> Dict[str, str]:
        """
        List all available models
        
        Returns:
            Dictionary mapping stock symbols to model paths
        """
        models = {}
        for model_file in self.models_dir.glob('*_model.h5'):
            symbol = model_file.stem.split('_')[0]
            models[symbol] = str(model_file)
        return models

    def delete_model(self, symbol: str) -> bool:
        """
        Delete a model file
        
        Args:
            symbol: Stock symbol associated with the model
            
        Returns:
            True if model was deleted, False if it didn't exist
        """
        model_path = self.models_dir / f'{symbol}_model.h5'
        if model_path.exists():
            model_path.unlink()
            return True
        return False
