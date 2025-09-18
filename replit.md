# Vehicle Detection and Counting System

## Overview
This is a real-time vehicle detection and counting system built with Python, OpenCV, and Flask. The application detects and counts vehicles using computer vision techniques and provides a web-based interface for monitoring.

## Current State
- ✅ Successfully converted from desktop OpenCV application to web-based Flask application
- ✅ Working demo mode with simulated vehicle detection when camera is not available
- ✅ Web interface running on port 5000
- ✅ Real-time vehicle counting with statistics display
- ✅ Flask server properly configured with CORS support
- ✅ Deployment configured for autoscale
- ✅ **Replit environment setup complete (Sept 18, 2025)**
- ✅ **Python 3.11 module installed with all required dependencies**
- ✅ **HTML templates created for professional web interface**
- ✅ **Code issues resolved and application running successfully**
- ✅ **Workflow configured to serve on port 5000 with host 0.0.0.0**
- ✅ **Enhanced dashboard with modern UI and responsive design (Sept 18, 2025)**
- ✅ **All API endpoints implemented for frontend interaction**
- ✅ **Multi-lane video feeds working properly**
- ✅ **Traffic signal control system operational**

## Project Architecture
- **Backend**: Flask web server with OpenCV processing
- **Frontend**: HTML/CSS/JavaScript interface with live video streaming
- **Computer Vision**: MOG2 background subtraction for vehicle detection
- **Demo Mode**: Synthetic vehicle simulation when camera unavailable

## Recent Changes (Sept 18, 2025)
- ✅ Evolved from single-lane to **multi-lane traffic management system**
- ✅ Built **professional traffic controller dashboard** with manual overrides
- ✅ Implemented **intelligent traffic signal control** based on traffic density analysis
- ✅ Added **real traffic camera video feeds** (ambulance route, Tilton traffic, general traffic)
- ✅ Fixed critical bugs: JSON serialization, traffic density calculations, manual override conflicts
- ✅ Enhanced frontend with robust error handling and real-time updates
- ✅ **Professional dashboard interface with Inter font and modern design**
- ✅ **Responsive layout optimized for 90% zoom in Chrome**
- ✅ **Complete API endpoints for traffic control and monitoring**
- ✅ **Real-time lane status updates and signal control**
- ✅ **Emergency controls and system reset functionality**

## Technical Features
- **Multi-Lane Processing**: Simultaneous processing of 3 traffic camera feeds
- **Intelligent Traffic Analysis**: Rolling 5-second density calculations with automatic signal control
- **Professional Controller Dashboard**: Traffic management interface with manual/auto modes
- **Smart Signal Control**: Automatically stops low-traffic lanes, prioritizes high-traffic areas
- **Manual Override System**: Individual lane control with emergency override capabilities
- **Real-time Video Streaming**: Combined multi-camera view at ~30 FPS
- **REST API Suite**: Complete control endpoints for signal management and statistics
- **Responsive Design**: Professional traffic control center aesthetics

## Dependencies
- opencv-python==4.8.1.78
- numpy==1.24.3
- matplotlib==3.7.2
- flask==2.3.3
- flask-cors==4.0.0

## User Preferences
- Web-based interface preferred over desktop application
- Real-time streaming and statistics display
- Clean, professional UI with vehicle counting visualization