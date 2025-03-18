import os
from datetime import datetime
from typing import Dict, List, Optional
import googlemaps
from traffic_predictor import TrafficPredictor
import numpy as np

class RouteOptimizer:
    def __init__(self, traffic_predictor: TrafficPredictor):
        self.gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))
        self.traffic_predictor = traffic_predictor
        
    def get_optimal_route(self, origin: str, destination: str, 
                         departure_time: Optional[datetime] = None) -> Dict:
        """
        Find the optimal route between two points considering traffic predictions
        
        Args:
            origin: Starting location (address or lat,lng)
            destination: End location (address or lat,lng)
            departure_time: Optional departure time (defaults to now)
        
        Returns:
            Dictionary containing the optimal route information
        """
        if departure_time is None:
            departure_time = datetime.now()
            
        try:
            # Get alternative routes from Google Maps
            routes = self.gmaps.directions(
                origin,
                destination,
                alternatives=True,
                departure_time=departure_time,
                traffic_model='best_guess'
            )
            
            if not routes:
                raise ValueError("No routes found")
            
            # Score and rank each route
            scored_routes = []
            for route in routes:
                score = self._calculate_route_score(route, departure_time)
                scored_routes.append((score, route))
            
            # Get the best route (lowest score)
            best_score, best_route = min(scored_routes, key=lambda x: x[0])
            
            # Extract useful information from the route
            route_info = self._extract_route_info(best_route)
            route_info['traffic_score'] = best_score
            
            return route_info
            
        except Exception as e:
            raise Exception(f"Error finding optimal route: {str(e)}")
    
    def _calculate_route_score(self, route: Dict, departure_time: datetime) -> float:
        """Calculate a score for a route based on multiple factors"""
        duration = route['legs'][0]['duration']['value']  # in seconds
        distance = route['legs'][0]['distance']['value']  # in meters
        
        # Extract route segments for traffic prediction
        segments = self._extract_route_segments(route)
        
        # Get traffic predictions for each segment
        traffic_score = 0
        for segment in segments:
            # Create feature vector for traffic prediction
            # [hour, day_of_week, latitude, longitude]
            features = np.array([
                departure_time.hour,
                departure_time.weekday(),
                segment['lat'],
                segment['lng']
            ]).reshape(1, 4)
            
            # Get traffic prediction for this segment
            segment_traffic = self.traffic_predictor.predict_traffic(features)
            traffic_score += segment_traffic * segment['distance']
        
        # Normalize traffic score by total distance
        traffic_score = traffic_score / distance if distance > 0 else 0
        
        # Calculate final score (weighted combination of factors)
        weights = {
            'duration': 0.4,
            'traffic': 0.4,
            'distance': 0.2
        }
        
        normalized_duration = duration / 3600  # convert to hours
        normalized_distance = distance / 1000  # convert to kilometers
        
        final_score = (
            weights['duration'] * normalized_duration +
            weights['traffic'] * traffic_score +
            weights['distance'] * normalized_distance
        )
        
        return final_score
    
    def _extract_route_segments(self, route: Dict) -> List[Dict]:
        """Extract key points along the route for traffic prediction"""
        segments = []
        
        for step in route['legs'][0]['steps']:
            segment = {
                'lat': step['start_location']['lat'],
                'lng': step['start_location']['lng'],
                'distance': step['distance']['value']
            }
            segments.append(segment)
        
        # Add final destination
        segments.append({
            'lat': route['legs'][0]['end_location']['lat'],
            'lng': route['legs'][0]['end_location']['lng'],
            'distance': 0
        })
        
        return segments
    
    def _extract_route_info(self, route: Dict) -> Dict:
        """Extract useful information from a route"""
        leg = route['legs'][0]
        
        return {
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
                    'instructions': step['html_instructions']
                }
                for step in leg['steps']
            ]
        }
