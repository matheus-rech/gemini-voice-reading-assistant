import React, { useState } from 'react';

function MicrophoneControl({ isRecording, onToggleRecording }) {
  return (
    <button 
      onClick={onToggleRecording}
      className={`mic-button ${isRecording ? 'recording' : ''}`}
      style={{
        padding: '10px',
        borderRadius: '50%',
        width: '50px',
        height: '50px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        border: 'none',
        backgroundColor: isRecording ? '#EA4335' : '#4285F4',
        color: 'white',
        cursor: 'pointer',
        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
      }}
    >
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path 
          d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" 
          fill="currentColor"
        />
        <path 
          d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" 
          fill="currentColor"
        />
      </svg>
    </button>
  );
}

export default MicrophoneControl;
