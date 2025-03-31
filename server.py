import asyncio
import json
import os
import signal
import sys
import traceback
from typing import Set, Dict, Any, List

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedError

# Import our main app
from main import AudioLoop, MODEL, SYSTEM_PROMPT

# Set up basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('gemini-server')

# Global variables to track active connections
active_connections: Set[WebSocketServerProtocol] = set()
running = True

# Create a simplified version of the main app that works with WebSockets
class WebSocketAudioLoop(AudioLoop):
    def __init__(self, websocket, video_mode="none"):
        super().__init__(video_mode=video_mode)
        self.websocket = websocket
        self.message_queue = asyncio.Queue()
    
    # Override methods to work with WebSocket
    async def send_response(self, text):
        """Send a text response to the WebSocket client."""
        try:
            await self.websocket.send(json.dumps({
                "type": "text",
                "content": text
            }))
        except Exception as e:
            logger.error(f"Error sending response: {e}")
    
    async def process_text(self, text):
        """Process text input from the WebSocket client."""
        response = await asyncio.to_thread(
            self.chat_session.send_message, text
        )
        await self.send_response(response.text)
    
    async def process_image(self, image_data):
        """Process image data from the WebSocket client."""
        # Implementation would depend on how images are sent over WebSocket
        # This is a placeholder
        await self.send_response("Image processing not implemented yet")
        
    async def process_audio(self, audio_data):
        """Process audio data from the WebSocket client."""
        try:
            # In a real implementation, we would:
            # 1. Convert the base64 audio to binary data
            # 2. Use a speech-to-text service to convert it to text
            # 3. Process the text with Gemini
            # 4. Send the response back
            
            # For now, just respond with a placeholder text message
            await self.send_response("I heard your message. How can I help you?")
            
            # Also send a mock audio response
            # In a real implementation, this would be generated from text-to-speech
            mock_audio_base64 = "UklGRiQEAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAEAAD+/wIA/f8EAP3/BQD7/wgA+v8KAPP/EADw/xQA7f8YAOX/HQDj/yAA3v8lANr/KgDV/y8A0f8yAM3/NQDK/zgAxv87AMT/PAC//z8Au/9BALX/RACy/0YAr/9IALL/RACt/0oAov9LAJj/TwCY/1EAl/9QAJ3/TQCW/1AAk/9SAJv/TQCU/1EAiv9UAH7/WQCY/00Afv9ZAH3/WgB7/1sAiv9VAIT/VwSm/0sBfv9eAHv/X/Q8/3cMTf9LAJH/VgJt/2hLBf9ZABT/qAFY/3MABP+mABL/qwAB/7kA9P7QAPIczQDn/twA2P7uAMX++wC0/v8Atv79ALv++QDm/ckTnP4gB5b+LAe1/v8Ap/4EAZ3+CQGp/gIBu/75ALz++ADM/uoA3P7ZAPT+wgAJ/7EA//67ABj/owAS/6wAFP+pABz/oAAy/4cAQf91AF//VgBy/z8Ajf8mAKf/CgDD/+8A6v+4ABf/gACH/w0Azv+uACv/RwL7/uUA7P4LAw3/2wBC/5IAnP8jAO3/vwBG/10Atf/zAGP/XQGS/6MAzP9wAA0AIwBg/8cAgP+jAKz/dgDx/w0ACP/VAJX/gADn/zEAIwDJAHP/rAD2/y4APQCyAGv/uwAGAB8AXQCWAJj/fgIo/54Aj/8/Acn/ZADp/zcANgAdAVP/jwCo//j/WQBIAIL/ewE+/wYAif9ZALT/dwCg/2MApv9WAOv/FQCU/2cBQ/+HACv/kgDA/3MAs/9aALj/UwDC/0QA9v/n/5X/cgE2/6MAkP9mAaj/XwCs/2MAnf9qAKb/XACz/0kA3f/7/6r/VQFY/3IAzf8WARL/FAEE/yoA/P40ABL/JAAV/x8AGP8ZADj/9f9C/+v/R//m/1H/2v9b/8//Xv/M/27/uv+G/6X/mv+R/7X/d//N/13/5f9G//v/Nf8JACP/FQAQ/x4AB/8iAAL/JgD6/jAA8f42AOz+OwDk/kEA3/5FAOP+QQDl/j8A7f44APP+LwAA/yUA+f4wAPX+MQD3/i4A+/4qAAH/KQAE/ycACP8kAAv/IQAM/x8ADv8dABX/EwAf/wkAJ/8BAC3/+v8z//P/OP/v/z7/6v9G/+P/T//Z/1z/z/9t/8L/e/+4/4z/qP+f/5j/sP+K/8D/e//V/2b/6f9T//r/QP8JAC//FwAf/yEAE/8qAAn/MQAD/zQA//4zAPn+rTTp/jsA4P5AAOL+PQDl/joA6v42AO7+MwDy/jIA+v4sAAD/JgAE/yUABf8kAAb/IgAK/x4AEP8YABT/EwAX/w8AGf8NABX/EgAZ/w0AHP8JAB3/BwAe/wYAH/8EAB//BQAe/wYAHv8GACD/AQAo//f/Mf/s/zz/4f9J/9X/V//K/2X/vf9z/7D/gf+k/47/l/+c/4v/q/+A/7j/df/E/2v/zv9i/9f/W//d/1j/4P9W/+P/U//m/1D/6f9O/+r/Tf/q/03/6v9N/+r/T//o/1H/5v9T/+X/Vf/j/1j/4f9c/93/X//a/2P/1/9m/9X/af/T/2z/0P9w/83/c//L/3b/yP95/8b/e//E/33/wv9//8H/gP/A/4H/wP+B/8D/"
            
            # Send audio data back to client
            await self.websocket.send(json.dumps({
                "type": "audio",
                "content": mock_audio_base64
            }))
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await self.send_response(f"Error processing audio: {str(e)}")

async def handle_client(websocket: WebSocketServerProtocol, path: str):
    """Handle a WebSocket connection from a client."""
    global active_connections
    
    client_id = id(websocket)
    logger.info(f"Client connected: {client_id}")
    active_connections.add(websocket)
    
    # Create a client-specific instance of our app
    client_app = WebSocketAudioLoop(websocket)
    
    # Initialize chat session
    client_app.model.start_chat(history=[
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["I understand my role as a helpful voice assistant. I'm ready to help you with whatever you'd like to show me or ask me about."]}
    ])
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                await process_message(websocket, data, client_app)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "content": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                await websocket.send(json.dumps({
                    "type": "error",
                    "content": f"Server error: {str(e)}"
                }))
    except ConnectionClosedError:
        logger.info(f"Client {client_id} connection closed")
    finally:
        active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected")

async def process_message(websocket: WebSocketServerProtocol, data: Dict[str, Any], client_app: WebSocketAudioLoop):
    """Process a message from a client."""
    msg_type = data.get('type')
    content = data.get('content')
    
    if msg_type == 'text':
        # Process text message
        logger.info(f"Received text message: {content}")
        await client_app.process_text(content)
    
    elif msg_type == 'audio':
        # Process audio data
        logger.info("Received audio data")
        # Use our dedicated method for audio processing
        await client_app.process_audio(content)
    
    elif msg_type == 'mode':
        # Change the mode (camera, screen, none)
        new_mode = content
        logger.info(f"Changing mode to: {new_mode}")
        
        client_app.video_mode = new_mode
        await websocket.send(json.dumps({
            "type": "status",
            "content": f"Mode changed to {new_mode}"
        }))
    
    else:
        logger.warning(f"Unknown message type: {msg_type}")
        await websocket.send(json.dumps({
            "type": "error",
            "content": f"Unknown message type: {msg_type}"
        }))

async def broadcast_message(message: Dict[str, Any]):
    """Send a message to all connected clients."""
    if active_connections:
        await asyncio.gather(
            *[connection.send(json.dumps(message)) for connection in active_connections]
        )

async def start_server():
    """Start the WebSocket server."""
    global running
    
    # Configuration
    host = "localhost"
    port = 8000
    
    # Create and start the server
    logger.info(f"Starting WebSocket server on {host}:{port}")
    server = await websockets.serve(handle_client, host, port)
    
    # Keep the server running until stopped
    while running:
        await asyncio.sleep(1)
    
    # Clean up
    server.close()
    await server.wait_closed()
    logger.info("Server stopped")

def signal_handler(sig, frame):
    """Handle termination signals to clean up resources."""
    global running
    logger.info("Shutdown signal received")
    running = False

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the server
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    except Exception as e:
        logger.error(f"Server error: {e}")
        traceback.print_exc()
    finally:
        logger.info("Server shutdown complete")