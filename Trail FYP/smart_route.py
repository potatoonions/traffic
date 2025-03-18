import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import googlemaps
from dotenv import load_dotenv
import numpy as np
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Route Optimization System",
    description="AI-powered traffic prediction and route optimization system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))

class RouteRequest(BaseModel):
    origin: str
    destination: str
    departure_time: Optional[datetime] = None

class SimpleTrafficPredictor:
    def predict_traffic(self, time_of_day: int, day_of_week: int) -> float:
        """
        Simple rule-based traffic prediction
        Returns a traffic multiplier between 1.0 (no traffic) and 2.0 (heavy traffic)
        """
        # Peak hours: 7-9 AM and 5-7 PM on weekdays
        is_peak_hour = (7 <= time_of_day <= 9) or (17 <= time_of_day <= 19)
        is_weekday = 0 <= day_of_week <= 4
        
        if is_weekday and is_peak_hour:
            return 2.0  # Heavy traffic during peak hours
        elif is_weekday:
            return 1.5  # Moderate traffic during weekdays
        else:
            return 1.0  # Light traffic during weekends

class RouteOptimizer:
    def __init__(self):
        self.traffic_predictor = SimpleTrafficPredictor()
    
    def get_optimal_route(self, origin: str, destination: str, departure_time: Optional[datetime] = None) -> Dict:
        """Find the optimal route considering current traffic conditions"""
        if departure_time is None:
            departure_time = datetime.now()
        
        try:
            # Get alternative routes from Google Maps
            routes = gmaps.directions(
                origin,
                destination,
                alternatives=True,
                departure_time=departure_time,
                traffic_model='best_guess'
            )
            
            if not routes:
                raise ValueError("No routes found")
            
            # Score each route
            scored_routes = []
            for route in routes:
                score = self._calculate_route_score(route, departure_time)
                scored_routes.append((score, route))
            
            # Get the best route (lowest score)
            best_route = min(scored_routes, key=lambda x: x[0])[1]
            
            return self._format_route_response(best_route)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _calculate_route_score(self, route: Dict, departure_time: datetime) -> float:
        """Calculate a score for a route based on multiple factors"""
        leg = route['legs'][0]
        duration = leg['duration']['value']  # in seconds
        distance = leg['distance']['value']  # in meters
        
        # Get traffic multiplier based on time
        traffic_multiplier = self.traffic_predictor.predict_traffic(
            departure_time.hour,
            departure_time.weekday()
        )
        
        # Calculate weighted score (lower is better)
        duration_weight = 0.6
        distance_weight = 0.4
        
        normalized_duration = (duration * traffic_multiplier) / 3600  # convert to hours
        normalized_distance = distance / 1000  # convert to kilometers
        
        return (duration_weight * normalized_duration) + (distance_weight * normalized_distance)
    
    def _format_route_response(self, route: Dict) -> Dict:
        """Format the route information for API response"""
        leg = route['legs'][0]
        
        return {
            'summary': route.get('summary', ''),
            'duration': {
                'text': leg['duration']['text'],
                'value': leg['duration']['value']
            },
            'distance': {
                'text': leg['distance']['text'],
                'value': leg['distance']['value']
            },
            'start_address': leg['start_address'],
            'end_address': leg['end_address'],
            'steps': [
                {
                    'distance': step['distance']['text'],
                    'duration': step['duration']['text'],
                    'instructions': step['html_instructions'],
                    'travel_mode': step['travel_mode']
                }
                for step in leg['steps']
            ]
        }

# Initialize route optimizer
route_optimizer = RouteOptimizer()

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
    """Get the optimal route between two points"""
    return route_optimizer.get_optimal_route(
        route_request.origin,
        route_request.destination,
        route_request.departure_time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
