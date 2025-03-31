import React, { useState, useRef, useEffect } from 'react';

function ChatBox({ messages, onSend }) {
  const [message, setMessage] = useState('');
  const messagesEndRef = useRef(null);

  const handleSend = () => {
    if (message.trim()) {
      onSend(message);
      setMessage('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="chat-box">
      <div className="messages-container" style={{ height: '400px', overflow: 'auto', marginBottom: '10px', padding: '10px', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #dadce0' }}>
        {messages.map((msg, index) => (
          <div 
            key={index} 
            className={`message ${msg.sender}`}
            style={{
              padding: '8px 12px',
              borderRadius: '12px',
              marginBottom: '8px',
              maxWidth: '80%',
              wordBreak: 'break-word',
              backgroundColor: msg.sender === 'user' ? '#e8f0fe' : '#f8f9fa',
              alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              marginLeft: msg.sender === 'user' ? 'auto' : '0',
            }}
          >
            {msg.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="message-input" style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          style={{ flex: 1, padding: '10px', borderRadius: '24px', border: '1px solid #dadce0' }}
        />
        <button 
          onClick={handleSend}
          style={{ 
            padding: '10px 16px', 
            backgroundColor: '#4285F4', 
            color: 'white', 
            border: 'none', 
            borderRadius: '24px',
            cursor: 'pointer' 
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatBox;
