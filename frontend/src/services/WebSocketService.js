class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.messageHandler = null;
    this.statusChangeHandler = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimeout = null;
    this.pollingInterval = 5000; // 5 seconds
    this.pollingTimeout = null;
    this.token = null; // Token for authentication
  }

  connect() {
    // Check if already connected
    if (this.socket && this.isConnected) {
      console.log('WebSocket already connected');
      return;
    }

    // Close existing socket if any
    if (this.socket) {
      this.socket.close();
    }

    // Create new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.REACT_APP_WS_HOST || window.location.hostname;
    const port = process.env.REACT_APP_WS_PORT || '8000';
    const wsUrl = `${protocol}//${host}:${port}`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);
    
    try {
      this.socket = new WebSocket(wsUrl);

      // Set up event handlers
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  handleOpen() {
    console.log('WebSocket connected');
    this.isConnected = true;
    this.reconnectAttempts = 0;
    if (this.statusChangeHandler) {
      this.statusChangeHandler(true);
    }
  }

  handleClose(event) {
    console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
    this.isConnected = false;
    
    if (this.statusChangeHandler) {
      this.statusChangeHandler(false);
    }
    
    // Attempt to reconnect if not explicitly closed by the client
    if (event.code !== 1000) {
      this.scheduleReconnect();
    } else {
      // Switch to HTTP polling as a fallback
      this.startPolling();
    }
  }

  handleError(error) {
    console.error('WebSocket error:', error);
    // The WebSocket will attempt to reconnect automatically on error
    this.logError(error);
    this.displayUserFriendlyError(error);
  }

  handleMessage(event) {
    try {
      const data = JSON.parse(event.data);
      console.log('Received WebSocket message type:', data.type);
      
      if (this.messageHandler) {
        this.messageHandler(data);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
      
      console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
      
      // Clear any existing timeout
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
      }
      
      this.reconnectTimeout = setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, delay);
    } else {
      console.error('Maximum reconnect attempts reached');
      // Switch to HTTP polling as a fallback
      this.startPolling();
    }
  }

  startPolling() {
    console.log('Starting HTTP polling as fallback');
    this.pollingTimeout = setInterval(() => {
      this.pollServer();
    }, this.pollingInterval);
  }

  stopPolling() {
    if (this.pollingTimeout) {
      clearInterval(this.pollingTimeout);
      this.pollingTimeout = null;
    }
  }

  async pollServer() {
    try {
      const response = await fetch('/api/poll', {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });
      const data = await response.json();
      console.log('Received polling data:', data);
      
      if (this.messageHandler) {
        this.messageHandler(data);
      }
    } catch (error) {
      console.error('Error during HTTP polling:', error);
    }
  }

  sendMessage(type, content) {
    if (!this.socket || !this.isConnected) {
      console.error('Cannot send message, WebSocket not connected');
      return false;
    }

    try {
      const message = JSON.stringify({ type, content });
      this.socket.send(message);
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      return false;
    }
  }

  sendText(text) {
    return this.sendMessage('text', text);
  }

  sendAudio(audioData) {
    return this.sendMessage('audio', audioData);
  }

  sendMode(mode) {
    return this.sendMessage('mode', mode);
  }

  setMessageHandler(handler) {
    this.messageHandler = handler;
  }

  setStatusChangeHandler(handler) {
    this.statusChangeHandler = handler;
  }

  setToken(token) {
    this.token = token;
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.pollingTimeout) {
      clearInterval(this.pollingTimeout);
      this.pollingTimeout = null;
    }

    if (this.socket) {
      this.socket.close(1000, 'Client disconnected');
      this.socket = null;
    }
    
    this.isConnected = false;
  }

  logError(error) {
    // Implement detailed error logging
    console.error('Detailed WebSocket error:', error);
  }

  displayUserFriendlyError(error) {
    // Implement user-friendly error messages
    alert('An error occurred with the WebSocket connection. Please try again later.');
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();
export default webSocketService;
