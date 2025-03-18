# Smart Route Optimization System

An AI-powered traffic and route optimization system that uses real-time data to predict traffic patterns and suggest optimal routes.

## Features
- Traffic pattern prediction using LSTM models
- Real-time route optimization
- Integration with Google Maps API
- RESTful API endpoints for route optimization

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
- Copy `.env.example` to `.env`
- Add your Google Maps API key to `.env`

3. Run the application:
```bash
python main.py
```

## API Endpoints
- GET `/`: Health check
- GET `/optimize-route`: Get optimized route between two points
  - Parameters:
    - `origin`: Starting location
    - `destination`: End location
