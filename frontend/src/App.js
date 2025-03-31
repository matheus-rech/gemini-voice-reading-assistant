import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import VideoPreview from './components/VideoPreview';
import ChatBox from './components/ChatBox';
import MicrophoneControl from './components/MicrophoneControl';
import AudioRecorder from './components/AudioRecorder';
import webSocketService from './services/WebSocketService';

// Create a singleton AudioContext instance
let globalAudioContext = null;

function getAudioContext() {
  if (!globalAudioContext) {
    // Create the audio context only once
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      globalAudioContext = new AudioContext();
    } catch (e) {
      console.error('Failed to create AudioContext:', e);
    }
  }
  return globalAudioContext;
}

function App() {
  const [mode, setMode] = useState('camera'); // 'camera', 'screen', or 'none'
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const audioContextRef = useRef(null);

  // Initialize web socket and audio context
  useEffect(() => {
    // Set up WebSocket event handlers
    webSocketService.setMessageHandler(handleReceiveMessage);
    webSocketService.setStatusChangeHandler(handleStatusChange);
    
    // Connect to WebSocket server
    webSocketService.connect();
    
    // Get singleton AudioContext instance
    audioContextRef.current = getAudioContext();
    
    // Cleanup function
    return () => {
      webSocketService.disconnect();
    };
  }, []);

  // Handle mode changes
  useEffect(() => {
    // Notify server of mode change
    if (isConnected) {
      webSocketService.sendMode(mode);
    }
  }, [mode, isConnected]);

  // Handle received messages
  const handleReceiveMessage = (data) => {
    if (data.type === 'text') {
      setMessages(prev => [...prev, {
        sender: 'gemini',
        text: data.content
      }]);
    } else if (data.type === 'audio') {
      // Process audio data
      playAudio(data.content);
    } else if (data.type === 'error') {
      console.error('Error from server:', data.content);
      // Optionally display error to user
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `Error: ${data.content}`
      }]);
    }
  };

  // Handle status changes
  const handleStatusChange = (status) => {
    setIsConnected(status);
    setConnectionStatus(status ? 'Connected' : 'Disconnected');
  };

  // Send a text message
  const handleSendMessage = (text) => {
    // Add message to local state
    setMessages(prev => [...prev, {
      sender: 'user',
      text: text
    }]);

    // Send to server
    webSocketService.sendText(text);
  };

  // Toggle microphone recording
  const handleToggleRecording = () => {
    setIsRecording(prev => !prev);
  };

  // Manually trigger reconnection attempt
  const handleReconnect = () => {
    webSocketService.connect();
  };

  // Play audio data received from the server
  const playAudio = (audioData) => {
    if (!audioContextRef.current) {
      console.error('No audio context available');
      return;
    }
    
    try {
      console.log('Playing audio data...');
      
      // Decode the base64 audio data
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Make sure AudioContext is resumed (needed for browsers that suspend it)
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
      
      // Add to queue
      audioQueueRef.current.push(bytes.buffer);
      console.log('Added audio to queue. Queue length:', audioQueueRef.current.length);
      
      // If not already playing, start
      if (!isPlayingRef.current) {
        console.log('Starting audio playback');
        playNextInQueue();
      }
    } catch (error) {
      console.error('Error processing audio data:', error);
    }
  };

  // Play the next audio chunk in the queue
  const playNextInQueue = () => {
    if (!audioContextRef.current || audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      console.log('Audio queue empty or no audio context');
      return;
    }

    isPlayingRef.current = true;
    const nextAudio = audioQueueRef.current.shift();
    console.log('Processing next audio chunk');

    try {
      // Decode and play audio
      audioContextRef.current.decodeAudioData(nextAudio)
        .then(buffer => {
          console.log('Successfully decoded audio buffer');
          
          // Create audio source
          const source = audioContextRef.current.createBufferSource();
          source.buffer = buffer;
          
          // Create gain node for volume control
          const gainNode = audioContextRef.current.createGain();
          gainNode.gain.value = 1.0; // Full volume
          
          // Connect the nodes: source -> gain -> destination
          source.connect(gainNode);
          gainNode.connect(audioContextRef.current.destination);
          
          // Set up callback for when audio finishes
          source.onended = () => {
            console.log('Audio chunk finished playing');
            playNextInQueue();
          };
          
          // Start playing
          source.start();
          console.log('Started playing audio chunk');
        })
        .catch(err => {
          console.error('Error decoding audio:', err);
          // Try next chunk if there was an error
          playNextInQueue();
        });
    } catch (err) {
      console.error('Critical error in audio playback:', err);
      isPlayingRef.current = false;
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1 className="app-title">Gemini Voice Reading Assistant</h1>
        <h3 className="app-subtitle">Ask questions or show documents to Gemini AI</h3>
      </header>

      <div className="mode-selector">
        <button 
          className={`mode-button ${mode === 'camera' ? 'active' : ''}`}
          onClick={() => setMode('camera')}
        >
          Camera Mode
        </button>
        <button 
          className={`mode-button ${mode === 'screen' ? 'active' : ''}`}
          onClick={() => setMode('screen')}
        >
          Screen Share
        </button>
        <button 
          className={`mode-button ${mode === 'none' ? 'active' : ''}`}
          onClick={() => setMode('none')}
        >
          Audio Only
        </button>
      </div>

      <div className="main-content">
        <div className="video-chat-container">
          <div className="video-container">
            <VideoPreview mode={mode} />
          </div>
          
          <div className="chat-container">
            <ChatBox 
              messages={messages} 
              onSend={handleSendMessage} 
            />
          </div>
        </div>

        <div className="control-panel">
          <MicrophoneControl 
            isRecording={isRecording}
            onToggleRecording={handleToggleRecording}
          />
          <div style={{ color: isConnected ? '#34A853' : '#EA4335', margin: '10px' }}>
            {connectionStatus}
          </div>
          {!isConnected && (
            <button onClick={handleReconnect} style={{ padding: '10px', borderRadius: '24px', backgroundColor: '#4285F4', color: 'white', border: 'none', cursor: 'pointer' }}>
              Reconnect
            </button>
          )}
        </div>
      </div>
      
      {/* Invisible audio recorder component */}
      <AudioRecorder 
        isRecording={isRecording}
        onStatusChange={(status) => setIsRecording(status)}
        audioContext={audioContextRef.current}
      />
    </div>
  );
}

export default App;
