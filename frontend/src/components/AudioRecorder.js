import React, { useState, useEffect, useRef } from 'react';
import webSocketService from '../services/WebSocketService';

function AudioRecorder({ isRecording, onStatusChange, audioContext }) {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  
  useEffect(() => {
    // Start or stop recording based on isRecording prop
    if (isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
    
    // Cleanup function to ensure we stop recording and release media resources
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [isRecording]);

  const startRecording = async () => {
    audioChunksRef.current = [];
    
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      streamRef.current = stream;
      
      // Create MediaRecorder instance
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      // Set up event handlers
      mediaRecorder.ondataavailable = handleDataAvailable;
      mediaRecorder.onstop = handleRecordingStop;
      
      // Resume AudioContext if it's suspended (needed for Safari)
      if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      
      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms for real-time streaming
      console.log('Recording started');
      
      // Notify parent of successful start
      if (onStatusChange) {
        onStatusChange(true);
      }
    } catch (err) {
      console.error('Error starting recording:', err);
      if (onStatusChange) {
        onStatusChange(false);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      console.log('Recording stopped');
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  const handleDataAvailable = (event) => {
    if (event.data && event.data.size > 0) {
      audioChunksRef.current.push(event.data);
      
      // For real-time streaming, you might want to send each chunk as it becomes available
      sendAudioChunk(event.data);
    }
  };

  const handleRecordingStop = () => {
    // Convert all recorded chunks to a single blob
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    
    // You might want to send the complete recording here as well
    // or just rely on the chunks that were sent in real-time
    
    // Reset state
    audioChunksRef.current = [];
    
    // Notify parent
    if (onStatusChange) {
      onStatusChange(false);
    }
  };

  const sendAudioChunk = async (chunk) => {
    // Convert the blob to base64 for sending over WebSocket
    const reader = new FileReader();
    reader.readAsDataURL(chunk);
    
    reader.onloadend = () => {
      // Extract the base64 data (remove the data URL prefix)
      const base64data = reader.result.split(',')[1];
      
      // Send to server
      webSocketService.sendAudio(base64data);
    };
  };

  // Invisible component - no UI
  return null;
}

export default AudioRecorder;