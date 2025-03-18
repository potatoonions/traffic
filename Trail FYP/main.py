from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from traffic_predictor import TrafficPredictor
from route_optimizer import RouteOptimizer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Route Optimization System",
    description="AI-powered traffic prediction and route optimization system",
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
traffic_predictor = TrafficPredictor()
route_optimizer = RouteOptimizer(traffic_predictor)

class RouteRequest(BaseModel):
    origin: str
    destination: str
    departure_time: Optional[datetime] = None

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Smart Route Optimization System",
        "version": "1.0.0"
    }

@app.post("/optimize-route")
def optimize_route(route_request: RouteRequest):
    """
    Get the optimal route between two points
    
    Args:
        route_request: RouteRequest object containing origin and destination
        
    Returns:
        Optimized route information including traffic predictions
    """
    try:
        return route_optimizer.get_optimal_route(
            route_request.origin,
            route_request.destination,
            route_request.departure_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/traffic-prediction")
def get_traffic_prediction(latitude: float, longitude: float, time: Optional[datetime] = None):
    """
    Get traffic prediction for a specific location and time
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        time: Optional prediction time (defaults to current time)
        
    Returns:
        Traffic prediction score
    """
    try:
        if time is None:
            time = datetime.now()
            
        features = [
            time.hour,
            time.weekday(),
            latitude,
            longitude
        ]
        
        prediction = traffic_predictor.predict_traffic(features)
        return {
            "location": {"lat": latitude, "lng": longitude},
            "time": time.isoformat(),
            "traffic_prediction": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
