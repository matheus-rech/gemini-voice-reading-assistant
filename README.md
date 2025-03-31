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

## Development Environment Setup

To set up a development environment with necessary dependencies, you can use the provided devcontainer configuration.

1. Ensure you have Docker installed on your machine.
2. Open the project in Visual Studio Code.
3. Install the "Remote - Containers" extension in Visual Studio Code.
4. Open the Command Palette (Ctrl+Shift+P) and select "Remote-Containers: Open Folder in Container...".
5. Select the project folder. Visual Studio Code will build the devcontainer and open the project inside the container.

## Handling Potential Issues

### ALSA Library Issues

If you encounter errors related to the ALSA library, such as "cannot find card '0'" or "Invalid value for card", follow these steps:

1. **Check audio device availability**: Ensure that your system has a working audio device. You can list available audio devices using the `arecord -l` command. If no devices are listed, you may need to check your hardware connections or install the necessary drivers.

2. **Configure ALSA**: The errors suggest that ALSA cannot find the default audio device. You can create or update the `~/.asoundrc` file to specify the default audio device. Here is an example configuration:
   ```plaintext
   pcm.!default {
       type hw
       card 0
   }

   ctl.!default {
       type hw
       card 0
   }
   ```
   Replace `card 0` with the appropriate card number for your system.

3. **Check PulseAudio**: The errors also indicate issues with PulseAudio. Ensure that PulseAudio is installed and running. You can start PulseAudio with the `pulseaudio --start` command. If PulseAudio is not installed, you can install it using your package manager (e.g., `sudo apt-get install pulseaudio`).

4. **Permissions**: Ensure that your user has the necessary permissions to access audio devices. You can add your user to the `audio` group with the following command:
   ```bash
   sudo usermod -aG audio $USER
   ```
   After running this command, log out and log back in for the changes to take effect.

5. **Update `main.py`**: The `AudioLoop` class in `main.py` already includes error handling for audio stream initialization. Ensure that the error messages provide enough information to diagnose the issue. You can add more detailed logging if necessary.
