import os
import tensorflow as tf

class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, filename):
        """
        Save a TensorFlow model to disk
        
        Args:
            model: TensorFlow model to save
            filename: Name of the file to save the model
        """
        model_path = os.path.join(self.model_dir, filename)
        model.save(model_path)
        return {"status": "success", "message": f"Model saved to {model_path}"}

    def load_model(self, filename):
        """
        Load a TensorFlow model from disk
        
        Args:
            filename: Name of the file containing the model
            
        Returns:
            Loaded TensorFlow model
        """
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return tf.keras.models.load_model(model_path)

    def list_models(self):
        """
        List all available models in the model directory
        
        Returns:
            List of model filenames
        """
        return [f for f in os.listdir(self.model_dir) if f.endswith('.h5')]

    def delete_model(self, filename):
        """
        Delete a model file
        
        Args:
            filename: Name of the file to delete
        """
        model_path = os.path.join(self.model_dir, filename)
        if os.path.exists(model_path):
            os.remove(model_path)
            return {"status": "success", "message": f"Model deleted: {filename}"}
        else:
            raise FileNotFoundError(f"Model file not found: {filename}")
