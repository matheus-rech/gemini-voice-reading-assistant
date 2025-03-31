import asyncio
import os
import sys
import time
import traceback
import hashlib
import argparse
from typing import List, Optional, Tuple, Dict, Any, Union

import cv2
from google import genai
from google.genai import types
import mss
import numpy as np
import pyaudio
from PIL import Image, ImageEnhance, ImageFilter

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.0-flash-exp"
DEFAULT_MODE = "camera"
CAPTURE_INTERVAL_MIN = 0.05  # min time between captures (seconds)
CAPTURE_INTERVAL_MAX = 0.5   # max time between captures (seconds)
DEFAULT_CAPTURE_INTERVAL = 0.2  # initial capture interval

# API key management
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize Gemini client with API key
client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=API_KEY)

# Configure audio response settings
CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "audio",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

# Initialize PyAudio
pya = pyaudio.PyAudio()

# User prompt that defines assistant behavior and constraints
SYSTEM_PROMPT = """
You are a helpful AI voice assistant. You can see what the user is showing you
through their camera or screen and respond to their questions about what you see.

For text you can see:
- If the user shows you a document or text on their screen, read it when asked.
- Summarize longer text content when appropriate.
- If asked to read something specific in the image, focus on that text.
- When reading text, present it in a logical order (top-to-bottom, left-to-right).
- For structured content like tables or forms, describe the structure first, then the content.

For images:
- Describe what you see when asked.
- Answer questions about objects, people, or scenes in the image.
- Be concise in your responses unless the user asks for more detail.

Text recognition commands:
- "Read this" or "What does it say?" - Read the text visible on screen
- "Summarize this page" - Provide a concise summary of visible text
- "Find [X] on this page" - Locate specific information in visible text

General guidelines:
- Be helpful, accurate, and concise in your responses.
- If you can't see something clearly, it's okay to say so.
- Don't make up information if you're uncertain.
- Respond conversationally and naturally.
- If the user asks you to remember something, do your best to keep it in mind.
"""

class ScreenCapture:
    """Enhanced screen capture with text recognition optimization."""
    
    def __init__(self, 
                 capture_interval: float = 0.2,
                 min_interval: float = 0.05,
                 max_interval: float = 0.5,
                 max_width: int = 1280):
        """
        Initialize screen capture with configurable parameters.
        
        Args:
            capture_interval: Initial interval between captures (seconds)
            min_interval: Minimum capture interval (seconds)
            max_interval: Maximum capture interval (seconds)
            max_width: Maximum width for resizing large captures
        """
        self.capture_interval = capture_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_width = max_width
        self.last_capture_time = 0
        self.last_image_hash = None
        self.text_region_cache = {}
        self.performance_metrics = {
            "capture_times": [],
            "processing_times": [],
            "transmission_times": [],
            "response_times": []
        }
    
    def adjust_capture_interval(self, avg_latency: float) -> None:
        """
        Adjust capture interval based on latency feedback.
        
        Args:
            avg_latency: Average latency from recent captures (seconds)
        """
        if avg_latency > 0.7:
            # If latency is high, slow down capture rate
            self.capture_interval = min(self.max_interval, self.capture_interval * 1.2)
        elif avg_latency < 0.3:
            # If latency is low, speed up capture rate
            self.capture_interval = max(self.min_interval, self.capture_interval * 0.9)
    
    async def capture_screen(self, monitor_num: int = 1) -> Optional[Image.Image]:
        """
        Capture the screen and return as PIL Image.
        
        Args:
            monitor_num: Monitor number to capture (default: primary)
            
        Returns:
            PIL Image of the captured screen or None if skipped
        """
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.capture_interval:
            return None
            
        capture_start = time.time()
        
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[monitor_num]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                
                # Resize if too large
                if img.width > self.max_width:
                    ratio = self.max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((self.max_width, new_height))
                
                # Log capture time
                capture_time = time.time() - capture_start
                self.performance_metrics["capture_times"].append(capture_time)
                
                # Update last capture time
                self.last_capture_time = current_time
                
                # Check if screen has changed significantly
                img_array = np.array(img.resize((32, 32)))
                img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
                
                if img_hash == self.last_image_hash:
                    # Screen hasn't changed, capture less frequently
                    self.capture_interval = min(self.max_interval, self.capture_interval * 1.1)
                    return None
                    
                self.last_image_hash = img_hash
                return img
                
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    def enhance_text_visibility(self, image: Image.Image) -> Image.Image:
        """
        Enhance image to improve text recognition.
        
        Args:
            image: Original PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        process_start = time.time()
        
        try:
            # Convert to OpenCV format for processing
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply slight blur to remove noise
            blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
            
            # Sharpen the image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            # Convert back to PIL image
            enhanced = Image.fromarray(sharpened)
            
            # Log processing time
            process_time = time.time() - process_start
            self.performance_metrics["processing_times"].append(process_time)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing text visibility: {e}")
            # Return original if enhancement fails
            return image
    
    async def detect_text_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions in the image that likely contain text.
        
        Args:
            image: PIL Image
            
        Returns:
            List of (x, y, width, height) regions
        """
        # Check if we have cached results for a similar image
        img_array = np.array(image.resize((32, 32)))
        img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        
        if img_hash in self.text_region_cache:
            return self.text_region_cache[img_hash]
            
        try:
            # Convert to OpenCV format
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Try using MSER for text detection
            try:
                mser = cv2.MSER_create()
                regions, _ = mser.detectRegions(gray)
                
                # Filter and merge text regions
                text_regions = []
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                    
                    # Apply heuristics to identify potential text regions
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.1 < aspect_ratio < 15 and 10 < w < image.width * 0.8 and 8 < h < image.height * 0.2:
                        # Likely a text region
                        text_regions.append((x, y, w, h))
            except Exception:
                # Fallback to simpler edge detection if MSER fails
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                text_regions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter using heuristics
                    if 0.1 < w/h < 10 and w > 20 and h > 10:
                        text_regions.append((x, y, w, h))
            
            # Merge overlapping regions
            text_regions = self._merge_overlapping_regions(text_regions)
            
            # Cache the results
            self.text_region_cache[img_hash] = text_regions
            
            # Limit cache size
            if len(self.text_region_cache) > 20:
                # Remove oldest entry
                oldest_key = next(iter(self.text_region_cache))
                del self.text_region_cache[oldest_key]
                
            return text_regions
            
        except Exception as e:
            print(f"Error detecting text regions: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping text regions.
        
        Args:
            regions: List of (x, y, width, height) regions
            
        Returns:
            List of merged regions
        """
        if not regions:
            return []
            
        # Sort by x-coordinate
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged_regions = [sorted_regions[0]]
        
        for current in sorted_regions[1:]:
            previous = merged_regions[-1]
            
            # Calculate overlap
            prev_x2 = previous[0] + previous[2]
            prev_y2 = previous[1] + previous[3]
            curr_x2 = current[0] + current[2]
            curr_y2 = current[1] + current[3]
            
            # Check for horizontal overlap
            h_overlap = (current[0] <= prev_x2) and (prev_x2 <= curr_x2)
            
            # Check for vertical overlap
            v_overlap = not ((current[1] >= prev_y2) or (curr_y2 <= previous[1]))
            
            if h_overlap and v_overlap:
                # Regions overlap, merge them
                x = min(previous[0], current[0])
                y = min(previous[1], current[1])
                w = max(prev_x2, curr_x2) - x
                h = max(prev_y2, curr_y2) - y
                
                merged_regions[-1] = (x, y, w, h)
            else:
                # No overlap, add current region
                merged_regions.append(current)
                
        return merged_regions
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get average performance metrics.
        
        Returns:
            Dictionary of average metric values
        """
        metrics = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                # Calculate average of last 10 values
                metrics[f"avg_{key}"] = sum(values[-10:]) / min(len(values), 10)
            else:
                metrics[f"avg_{key}"] = 0
                
        return metrics


class AudioLoop:
    def __init__(self, video_mode: str = DEFAULT_MODE):
        """Initialize the audio loop with specified video mode."""
        self.video_mode = video_mode
        self.running = True
        self.audio_stream = None
        try:
            self.audio_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
        except OSError as e:
            print(f"Error initializing audio stream: {e}")
        
        self.session = None
        self.audio_latencies: List[float] = []
        self.capture_interval = DEFAULT_CAPTURE_INTERVAL
        self.audio_in_queue = None
        self.last_capture_time = 0
        self.send_text_task = None
        
        # Initialize enhanced screen capture
        self.screen_capture = ScreenCapture(
            capture_interval=DEFAULT_CAPTURE_INTERVAL,
            min_interval=CAPTURE_INTERVAL_MIN,
            max_interval=CAPTURE_INTERVAL_MAX
        )
        
        # Store selected screen region (x, y, width, height) or None for full screen
        self.selected_region = None
        
        # Track command context
        self.last_command_context = {}

        # Initialize Gemini session with LiveConnect for audio
        self.connect = None
        self.chat_session = None

    async def initialize_gemini_session(self):
        """Initialize Gemini LiveConnect session with audio support."""
        try:
            # Initialize LiveConnect session with Gemini 2.0 Flash
            self.connect = await asyncio.to_thread(
                client.generate.live_connect, 
                model=MODEL,
                config=CONFIG
            )
            
            # Send initial system prompt
            response = await asyncio.to_thread(
                self.connect.send,
                SYSTEM_PROMPT
            )
            
            print("\nGemini: Ready to help with what you see!")
            
        except Exception as e:
            print(f"Error initializing Gemini session: {e}")
            traceback.print_exc()
            raise

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
                if self.connect:
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Create part with image MIME type
                    part = types.Part.blob(img_bytes, 'image/png')
                    
                    # Send to Gemini
                    response = await asyncio.to_thread(
                        self.connect.send,
                        part
                    )
                    
                    # Print text response
                    for chunk in response:
                        if chunk.text:
                            print("\nGemini: ", chunk.text)
                        
                        # Audio is handled automatically by LiveConnect

        except Exception as e:
            print(f"Camera capture error: {e}")
            traceback.print_exc()
        finally:
            if 'cap' in locals():
                cap.release()

    async def get_screen(self):
        """Enhanced screen capture with text recognition optimization."""
        try:
            while self.running:
                # Capture screen with adaptive interval
                img = await self.screen_capture.capture_screen()
                
                if img is None:
                    # No capture or no significant change
                    await asyncio.sleep(0.01)
                    continue
                
                # Process selected region if specified
                if self.selected_region:
                    x, y, width, height = self.selected_region
                    img = img.crop((x, y, x + width, y + height))
                
                # Check for recent text reading commands
                is_text_focused = False
                if self.last_command_context.get('command_type') == 'read_text':
                    is_text_focused = True
                    # Clear the context after using it
                    if time.time() - self.last_command_context.get('timestamp', 0) > 10:
                        self.last_command_context = {}
                
                # Detect text regions for text-focused mode
                text_regions = []
                if is_text_focused:
                    text_regions = await self.screen_capture.detect_text_regions(img)
                
                # Get enhanced version optimized for text recognition
                enhanced_img = None
                if is_text_focused or 'read' in self.last_command_context.get('command', '').lower():
                    enhanced_img = self.screen_capture.enhance_text_visibility(img)
                
                # Track capture and processing time
                capture_time = time.time()
                
                if self.connect:
                    # Convert images to bytes and send to Gemini
                    if is_text_focused and enhanced_img:
                        # Prepare text prompt and enhanced image for text reading
                        prompt = "This is a screen capture that contains text. I've processed it to make the text more readable. Please focus on reading and interpreting any text visible in the image."
                        
                        # Send text prompt first
                        await asyncio.to_thread(
                            self.connect.send,
                            prompt
                        )
                        
                        # Convert enhanced image to bytes
                        img_byte_arr = io.BytesIO()
                        enhanced_img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Create part with image MIME type
                        part = types.Part.blob(img_bytes, 'image/png')
                        
                        # Send enhanced image
                        response = await asyncio.to_thread(
                            self.connect.send,
                            part
                        )
                        
                    elif enhanced_img:
                        # Send both original and enhanced images with explanation
                        prompt = "This is a screen capture. I'm including both the original image and an enhanced version to help you read any text. Please focus on any text content when responding."
                        
                        # Send text prompt first
                        await asyncio.to_thread(
                            self.connect.send,
                            prompt
                        )
                        
                        # Send original image
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        img_part = types.Part.blob(img_bytes, 'image/png')
                        
                        await asyncio.to_thread(
                            self.connect.send,
                            img_part
                        )
                        
                        # Send enhanced image
                        enhanced_byte_arr = io.BytesIO()
                        enhanced_img.save(enhanced_byte_arr, format='PNG')
                        enhanced_bytes = enhanced_byte_arr.getvalue()
                        enhanced_part = types.Part.blob(enhanced_bytes, 'image/png')
                        
                        response = await asyncio.to_thread(
                            self.connect.send,
                            enhanced_part
                        )
                        
                    else:
                        # Standard image capture
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        part = types.Part.blob(img_bytes, 'image/png')
                        
                        response = await asyncio.to_thread(
                            self.connect.send,
                            part
                        )
                    
                    # Print text response
                    response_text = ""
                    for chunk in response:
                        if chunk.text:
                            response_text += chunk.text
                    
                    if response_text:
                        print("\nGemini: ", response_text)
                    
                    # Calculate response time
                    response_time = time.time() - capture_time
                    
                    # Update performance metrics
                    self.screen_capture.performance_metrics["response_times"].append(response_time)
                    self.audio_latencies.append(response_time)
                    
                    # Adjust capture interval based on performance
                    metrics = self.screen_capture.get_performance_metrics()
                    self.capture_interval = self.screen_capture.capture_interval
                    
        except Exception as e:
            print(f"Enhanced screen capture error: {e}")
            traceback.print_exc()

    # ---------------------------
    # Text Input Handling
    # ---------------------------
    async def send_text(self):
        """Handle user text input."""
        try:
            # Initialize Gemini LiveConnect session
            await self.initialize_gemini_session()
            
            print("\nGemini Voice Assistant running. Enter 'q' to quit.")
            print("Speak or type your question (or 'q' to quit):\n")
            
            while self.running:
                user_input = await asyncio.to_thread(input, "")
                
                if user_input.lower() == 'q':
                    self.running = False
                    raise asyncio.CancelledError("User requested quit")
                
                if user_input:
                    print(f"\nYou typed: {user_input}")
                    
                    # Check for screen reading commands
                    lower_input = user_input.lower()
                    if any(cmd in lower_input for cmd in ['read this', 'read the text', 'what does it say']):
                        # Mark this as a text reading command
                        self.last_command_context = {
                            'command': user_input,
                            'command_type': 'read_text',
                            'timestamp': time.time()
                        }
                        
                        # For text reading commands, don't need to send text separately
                        # The next screen capture will include the enhanced text processing
                        print("\nFocusing on text in next screen capture...")
                        continue
                    
                    # For region selection commands
                    if 'focus on region' in lower_input:
                        print("\nType the coordinates as x,y,width,height or 'reset' to clear region:")
                        region_input = await asyncio.to_thread(input, "")
                        
                        if region_input.lower() == 'reset':
                            self.selected_region = None
                            print("Cleared region selection. Using full screen.")
                        else:
                            try:
                                coords = [int(x.strip()) for x in region_input.split(',')]
                                if len(coords) == 4:
                                    self.selected_region = tuple(coords)
                                    print(f"Set region to {self.selected_region}")
                                else:
                                    print("Invalid format. Expected x,y,width,height")
                            except ValueError:
                                print("Invalid coordinates. Expected integers.")
                        continue
                    
                    # Send text to Gemini
                    if self.connect:
                        response = await asyncio.to_thread(
                            self.connect.send,
                            user_input
                        )
                        
                        # Print text response
                        response_text = ""
                        for chunk in response:
                            if chunk.text:
                                response_text += chunk.text
                        
                        if response_text:
                            print("\nGemini: ", response_text)
                        
        except EOFError:
            self.running = False
            raise asyncio.CancelledError("EOF on input")

    # ---------------------------
    # Audio Handling (Simplified for compatibility)
    # ---------------------------
    async def send_realtime(self):
        """Stream audio to Gemini."""
        try:
            # Buffer for collecting audio chunks
            audio_buffer = []
            is_speaking = False
            silence_count = 0
            
            # Define silence threshold
            SILENCE_THRESHOLD = 500  # Adjust based on your microphone
            
            while self.running:
                if self.audio_stream:
                    # Read audio chunk
                    try:
                        data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Convert to numpy array
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        
                        # Check if speaking (basic voice activity detection)
                        volume = np.abs(audio_array).mean()
                        
                        if volume > SILENCE_THRESHOLD:
                            # Reset silence counter when sound detected
                            silence_count = 0
                            
                            if not is_speaking:
                                is_speaking = True
                                print("\nListening...")
                            
                            # Add to buffer
                            audio_buffer.append(data)
                            
                        elif is_speaking:
                            # Still add some silence for natural pauses
                            audio_buffer.append(data)
                            silence_count += 1
                            
                            # If silence persists, assume speaking ended
                            if silence_count > 20:  # About 0.5 seconds of silence
                                is_speaking = False
                                
                                # Process collected audio
                                if audio_buffer and self.connect:
                                    print("Processing audio...")
                                    
                                    # Combine audio chunks
                                    audio_data = b''.join(audio_buffer)
                                    
                                    # LiveConnect supports audio input
                                    audio_part = types.Part.audio(audio_data, mime_type="audio/raw;encoding=signed-integer;bits_per_sample=16;sample_rate=16000")
                                    
                                    # Send audio to Gemini
                                    response = await asyncio.to_thread(
                                        self.connect.send,
                                        audio_part
                                    )
                                    
                                    # Print text response
                                    response_text = ""
                                    for chunk in response:
                                        if chunk.text:
                                            response_text += chunk.text
                                    
                                    if response_text:
                                        print("\nGemini: ", response_text)
                                
                                # Clear buffer for next utterance
                                audio_buffer = []
                                
                    except Exception as e:
                        print(f"Audio reading error: {e}")
                
                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Audio streaming error: {e}")
    
    async def listen_audio(self):
        """Monitoring for audio processing."""
        # Most audio handling is now in send_realtime
        while self.running:
            await asyncio.sleep(1.0)
    
    async def receive_audio(self):
        """
        Audio responses are handled automatically by LiveConnect.
        This is a placeholder for compatibility.
        """
        while self.running:
            await asyncio.sleep(1.0)
    
    async def play_audio(self):
        """
        Audio playback is handled automatically by LiveConnect.
        This is a placeholder for compatibility.
        """
        self.audio_in_queue = asyncio.Queue()  # For compatibility
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
                
                # Get screen capture metrics
                if self.video_mode == "screen":
                    metrics = self.screen_capture.get_performance_metrics()
                    metrics_str = ", ".join([f"{k}: {v:.3f}s" for k, v in metrics.items()])
                    print(f"\n[Performance] Avg latency: {avg_latency:.2f}s, {metrics_str}\n")
                else:
                    print(
                        f"\n[Performance] Avg audio latency: {avg_latency:.2f}s, "
                        f"capture_interval: {self.capture_interval:.2f}s\n"
                    )
                
                # Adjust interval based on latency
                self.screen_capture.adjust_capture_interval(avg_latency)
                
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
                
                # Clear old latency data
                if len(self.audio_latencies) > 50:
                    self.audio_latencies = self.audio_latencies[-20:]

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
            
            # Close Gemini LiveConnect session
            if self.connect:
                try:
                    self.connect.close()
                except Exception:
                    pass
                    
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
