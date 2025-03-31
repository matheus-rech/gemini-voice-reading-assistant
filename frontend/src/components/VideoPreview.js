import React, { useEffect, useRef } from 'react';

function VideoPreview({ mode }) {
  const videoRef = useRef(null);
  
  useEffect(() => {
    let stream = null;

    const startCamera = async () => {
      try {
        if (mode === 'camera') {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
        } else if (mode === 'screen') {
          stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        } else {
          return; // No video for 'none' mode
        }
        
        // Use ref to attach stream to video element
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing media devices:', err);
      }
    };

    startCamera();

    // Cleanup function
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [mode]);

  if (mode === 'none') {
    return <div className="video-placeholder">Audio Only Mode</div>;
  }

  return (
    <div className="video-preview">
      <video
        ref={videoRef}
        autoPlay
        style={{ width: '100%', height: 'auto', borderRadius: '8px' }}
      />
    </div>
  );
}

export default VideoPreview;
