import argparse
import asyncio
import os
import sys
import time
import traceback
from typing import List, Optional, Union

import cv2
import google.generativeai as genai
import mss
import numpy as np
import pyaudio
from PIL import Image

# API key management
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set")
    sys.exit(1)
genai.configure(api_key=API_KEY)

# Constants
MODEL = "gemini-1.5-flash"
DEFAULT_MODE = "camera"
CHUNK_SIZE = 1024
RECEIVE_SAMPLE_RATE = 24000
SEND_SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CAPTURE_INTERVAL_MIN = 0.05  # min time between captures (seconds)
CAPTURE_INTERVAL_MAX = 0.5   # max time between captures (seconds)
DEFAULT_CAPTURE_INTERVAL = 0.2  # initial capture interval

# User prompt that defines assistant behavior and constraints
SYSTEM_PROMPT = """
You are a helpful AI voice assistant. You can see what the user is showing you
through their camera or screen and respond to their questions about what you see.

For text you can see:
- If the user shows you a document or text on their screen, read it when asked.
- Summarize longer text content when appropriate.
- If asked to read something specific in the image, focus on that text.

For images:
- Describe what you see when asked.
- Answer questions about objects, people, or scenes in the image.
- Be concise in your responses unless the user asks for more detail.

General guidelines:
- Be helpful, accurate, and concise in your responses.
- If you can't see something clearly, it's okay to say so.
- Don't make up information if you're uncertain.
- Respond conversationally and naturally.
- If the user asks you to remember something, do your best to keep it in mind.
"""

class AudioLoop:
    def __init__(self, video_mode: str = DEFAULT_MODE):
        """Initialize the audio loop with specified video mode."""
        self.video_mode = video_mode
        self.running = True
        self.p = pyaudio.PyAudio()
        try:
            self.audio_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
        except OSError as e:
            print(f"Error initializing audio stream: {e}")
            self.audio_stream = None
        self.session = None
        self.audio_latencies: List[float] = []
        self.capture_interval = DEFAULT_CAPTURE_INTERVAL
        self.audio_in_queue = None
        self.last_capture_time = 0
        self.send_text_task = None

        # Initialize Gemini model
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(MODEL, generation_config=generation_config)
        self.chat_session = None

    # ---------------------------
    # Camera and Screen Capture
    # ---------------------------
    async def get_frames(self):
        """Capture frames from webcam and send to Gemini."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return

            # Configure camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self.running:
                current_time = time.time()
                if current_time - self.last_capture_time < self.capture_interval:
                    await asyncio.sleep(0.01)  # Small pause to avoid busy-waiting
                    continue

                self.last_capture_time = current_time
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    continue

                # Convert to RGB (from BGR) and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Send to Gemini model
                if self.chat_session:
                    response = await asyncio.to_thread(
                        self.chat_session.send_message, img
                    )
                    print("\nGemini: ", response.text)

        except Exception as e:
            print(f"Camera capture error: {e}")
        finally:
            if 'cap' in locals():
                cap.release()

    async def get_screen(self):
        """Capture screen and send to Gemini."""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor

                while self.running:
                    current_time = time.time()
                    if current_time - self.last_capture_time < self.capture_interval:
                        await asyncio.sleep(0.01)  # Small pause to avoid busy-waiting
                        continue

                    self.last_capture_time = current_time
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    
                    # Resize if too large (reduces bandwidth and processing)
                    if img.width > 1280:
                        ratio = 1280 / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((1280, new_height))

                    # Send to Gemini model
                    if self.chat_session:
                        response = await asyncio.to_thread(
                            self.chat_session.send_message, img
                        )
                        print("\nGemini: ", response.text)
        except Exception as e:
            print(f"Screen capture error: {e}")

    # ---------------------------
    # Text Input Handling
    # ---------------------------
    async def send_text(self):
        """Handle user text input."""
        try:
            # Initialize chat session with system prompt
            self.chat_session = self.model.start_chat(history=[
                {"role": "user", "parts": [SYSTEM_PROMPT]},
                {"role": "model", "parts": ["I understand my role as a helpful voice assistant. I'm ready to help you with whatever you'd like to show me or ask me about."]}
            ])
            
            print("\nGemini Voice Assistant running. Enter 'q' to quit.")
            print("Speak or type your question (or 'q' to quit):\n")
            
            while self.running:
                user_input = await asyncio.to_thread(input, "")
                
                if user_input.lower() == 'q':
                    self.running = False
                    raise asyncio.CancelledError("User requested quit")
                
                if user_input:
                    print(f"\nYou typed: {user_input}")
                    response = await asyncio.to_thread(
                        self.chat_session.send_message, user_input
                    )
                    print("\nGemini: ", response.text)
        except EOFError:
            self.running = False
            raise asyncio.CancelledError("EOF on input")

    # ---------------------------
    # Audio Handling (Simplified for compatability)
    # ---------------------------
    async def send_realtime(self):
        """Simplified placeholder for real-time audio."""
        while self.running:
            await asyncio.sleep(1.0)
    
    async def listen_audio(self):
        """Simplified placeholder for audio processing."""
        while self.running:
            await asyncio.sleep(1.0)
    
    async def receive_audio(self):
        """Simplified placeholder for audio receiving."""
        while self.running:
            await asyncio.sleep(1.0)
    
    async def play_audio(self):
        """Simplified placeholder for audio playback."""
        self.audio_in_queue = asyncio.Queue()  # Still need this for compatibility
        while self.running:
            await asyncio.sleep(1.0)

    # ---------------------------
    # Performance Monitor
    # ---------------------------
    async def monitor_performance(self):
        """
        Periodically checks average audio latency and adjusts capture intervals
        to reduce system load if latency is high.
        """
        while self.running:
            await asyncio.sleep(10.0)  # every 10 seconds
            if self.audio_latencies:
                avg_latency = sum(self.audio_latencies) / len(self.audio_latencies)
                print(
                    f"\n[Performance] Avg audio latency: {avg_latency:.2f}s, "
                    f"capture_interval: {self.capture_interval:.2f}s\n"
                )
                # If avg latency is high, increase capture interval
                if (avg_latency > 0.7):
                    self.capture_interval = min(
                        CAPTURE_INTERVAL_MAX, self.capture_interval * 1.2
                    )
                # If avg latency is quite low, we can decrease interval
                elif avg_latency < 0.3:
                    self.capture_interval = max(
                        CAPTURE_INTERVAL_MIN, self.capture_interval * 0.9
                    )

    # ---------------------------
    # Main Run Method
    # ---------------------------
    async def run(self):
        try:
            async with asyncio.TaskGroup() as tg:
                # Start tasks
                self.send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Launch either camera or screen capture (or none)
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.monitor_performance())

                # Wait for user input task to complete (which might raise CancelledError on 'q')
                await self.send_text_task

        except asyncio.CancelledError as c:
            # Graceful exit on user request
            print(f"\nShutting down: {c}")
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            self.running = False
            if self.audio_stream is not None:
                self.audio_stream.close()
            print("Cleaned up resources. Goodbye!")

# ---------------------------
# CLI Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Type of capture: camera, screen, or none",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
