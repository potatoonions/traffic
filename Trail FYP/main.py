from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

from stock_predictor import StockPredictor
from stock_trainer import StockTrainer
from model_manager import ModelManager

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Prediction System",
    description="AI-powered stock market prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
stock_predictor = StockPredictor("models")
stock_trainer = StockTrainer()
model_manager = ModelManager("models")

class StockRequest(BaseModel):
    symbol: str
    date: Optional[datetime] = None

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Stock Market Prediction System",
        "version": "1.0.0"
    }

@app.post("/predict-stock")
def predict_stock(stock_request: StockRequest):
    """
    Get the predicted stock price for a specific symbol and date
    
    Args:
        stock_request: StockRequest object containing symbol and date
        
    Returns:
        Predicted stock price information with historical data
    """
    try:
        # Load historical data
        df = stock_predictor.load_data(stock_request.symbol)
        
        # Get last 30 days of data for visualization
        last_30_days = df.tail(30)
        
        # Predict price
        result = stock_predictor.predict(
            stock_request.symbol,
            stock_request.date
        )
        
        # Prepare response with historical data
        return {
            "symbol": stock_request.symbol,
            "predicted_price": result["predicted_price"],
            "prediction_date": result["prediction_date"],
            "historical_dates": last_30_days.index.strftime("%Y-%m-%d").tolist(),
            "historical_prices": last_30_days["Close"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
def train_model(symbol: str, epochs: int = 50):
    """
    Train a new model for a specific stock symbol
    
    Args:
        symbol: Stock symbol to train
        epochs: Number of training epochs (default: 50)
        
    Returns:
        Training status and model details
    """
    try:
        result = stock_trainer.train_model(symbol, epochs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-model")
def load_model():
    """
    Load the stock prediction model
    
    Returns:
        Loaded model information
    """
    try:
        return model_manager.load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-model")
def save_model():
    """
    Save the stock prediction model
    
    Returns:
        Saved model information
    """
    try:
        return model_manager.save_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-company")
def search_company(query: str):
    """
    Search for company information using a search query
    
    Args:
        query: Search term to look for company names or symbols
        
    Returns:
        List of matching companies with their symbols and names
    """
    try:
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': query,
            'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'bestMatches' not in data:
            return []
            
        results = []
        for match in data['bestMatches']:
            results.append({
                'symbol': match['1. symbol'],
                'name': match['2. name'],
                'type': match['3. type'],
                'region': match['4. region'],
                'matchScore': match['9. matchScore']
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
