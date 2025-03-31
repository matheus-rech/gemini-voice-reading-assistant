# Gemini Voice Reading Assistant

An interactive voice assistant powered by Google's Gemini AI that can see and respond to what you show through your camera or screen.

## Features

- Real-time audio interaction with Gemini
- Camera or screen capture modes
- Text input support
- Automatic performance monitoring and adjustment
- Voice responses from the AI
- Detailed error logging and user-friendly error messages
- Exponential backoff for WebSocket reconnection attempts
- Fallback mechanisms to switch to HTTP polling when WebSocket connection is unavailable
- Token-based authentication for WebSocket and HTTP polling requests
- Clear feedback to the user about connection status and limitations in functionality

## Project Structure

This project consists of two main parts:

1. **Backend** (Python): Handles the connection to Gemini API, audio processing, and video capture
2. **Frontend** (React): Provides a user-friendly web interface

## Backend Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Set your Gemini API key as an environment variable:
   ```
   export GEMINI_API_KEY=your_api_key_here
   ```

3. Run the backend:
   ```bash
   # Use webcam for visual input
   python main.py --mode camera

   # Use screen capture for visual input
   python main.py --mode screen

   # Audio-only mode (no visual input)
   python main.py --mode none
   ```

## Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open http://localhost:3000 in your browser

## Interacting with the Assistant

- Speak naturally or type your questions
- The assistant can read text from your screen or camera
- Ask about what it sees or have it help with reading documents

## Requirements

- Python 3.10+
- Google Gemini API key
- Working microphone and speakers
- Webcam (for camera mode)
- Node.js and npm (for frontend)
