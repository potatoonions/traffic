from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

from stock_predictor import StockPredictor
from stock_trainer import StockTrainer
from model_manager import ModelManager
from stock_trend_predictor import StockTrendPredictor

load_dotenv()

app = FastAPI(
    title="Stock Market Prediction System",
    description="AI-powered stock market prediction system",
    version="1.0.0"
)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
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
stock_trend_predictor = StockTrendPredictor()

# Create trend prediction model
stock_trend_predictor.create_model()

class StockRequest(BaseModel):
    symbol: str
    date: Optional[str] = None

class CompanySearchRequest(BaseModel):
    query: str

class CompanyInfo(BaseModel):
    symbol: str
    name: str
    type: str
    region: str
    matchScore: str

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
    Get the predicted stock price and trend for a specific symbol
    
    Args:
        stock_request: StockRequest object containing symbol
        
    Returns:
        Predicted stock price information with historical data and trend prediction
    """
    try:
        # Load historical data
        df = stock_predictor.load_data(stock_request.symbol)
        
        # Train or load model if not exists
        try:
            stock_predictor.load_model(stock_request.symbol)
        except:
            # Train new model if it doesn't exist
            stock_predictor.train_model(df, epochs=50)
            stock_predictor.save_model(stock_request.symbol)
        
        # Get last 30 days of data for visualization
        last_30_days = df.tail(30)
        
        # Predict price
        price_result = stock_predictor.predict(stock_request.symbol)
        
        # Predict trend
        trend_result = stock_trend_predictor.predict_trend(df)
        
        # Prepare response with historical data
        return {
            "symbol": stock_request.symbol,
            "predicted_price": price_result["predicted_price"],
            "prediction_date": price_result["prediction_date"],
            "trend_prediction": trend_result["trend"],
            "confidence": trend_result["confidence"],
            "historical_dates": last_30_days.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "historical_prices": last_30_days["Close"].tolist()
        }
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
        response.raise_for_status()
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

@app.post("/add-company")
def add_company(company_info: CompanyInfo):
    """
    Add a new company to the database
    
    Args:
        company_info: Company information including symbol and name
        
    Returns:
        Status of the operation
    """
    try:
        # Load data for the new company
        df = stock_predictor.load_data(company_info.symbol)
        
        # Train model for the new company
        stock_trainer.train_model(company_info.symbol, epochs=50)
        
        # Train trend prediction model
        stock_trend_predictor.train_model(df, epochs=50)
        
        return {
            "status": "success",
            "message": f"Company {company_info.name} added successfully",
            "symbol": company_info.symbol
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
