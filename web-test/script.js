// HAILEI Chat Interface JavaScript

class HAILEIChat {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8002';
        this.websocket = null;
        this.sessionId = null;
        this.isConnected = false;
        this.currentAgent = 'coordinator';
        
        this.initializeElements();
        this.setupEventListeners();
        this.updateStatus('offline', 'Ready to start');
    }

    initializeElements() {
        // Form elements
        this.setupContainer = document.getElementById('setup-container');
        this.courseForm = document.getElementById('course-form');
        this.courseTitleInput = document.getElementById('course-title');
        this.courseLevelInput = document.getElementById('course-level');
        this.courseDurationInput = document.getElementById('course-duration');
        this.courseDescriptionInput = document.getElementById('course-description');
        
        // Chat elements
        this.chatContainer = document.getElementById('chat-container');
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.quickActions = document.getElementById('quick-actions');
        
        // Status elements
        this.statusDot = document.querySelector('.status-dot');
        this.statusText = document.getElementById('status-text');
        this.activeAgentSpan = document.getElementById('active-agent');
        this.agentRoleSpan = document.getElementById('agent-role');
        this.sessionDisplay = document.getElementById('session-display');
        this.connectionStatus = document.getElementById('connection-status');
        this.connectionMessage = document.getElementById('connection-message');
    }

    setupEventListeners() {
        // Course form submission
        this.courseForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startSession();
        });

        // Message input
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Send button
        this.sendBtn.addEventListener('click', () => {
            this.sendMessage();
        });

        // Auto-focus message input when typing
        document.addEventListener('keydown', (e) => {
            if (this.chatContainer.style.display !== 'none' && 
                !e.target.matches('input, textarea, button') && 
                e.key.length === 1) {
                this.messageInput.focus();
            }
        });
    }

    updateStatus(status, message) {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = message;
    }

    updateAgent(agentName, role) {
        this.currentAgent = agentName;
        this.activeAgentSpan.textContent = agentName;
        this.agentRoleSpan.textContent = role;
        
        // Update agent icon based on type
        const agentIcon = document.querySelector('.agent-icon');
        const icons = {
            'coordinator': '🎯',
            'ipdai': '📋',
            'cauthai': '✍️',
            'tfdai': '⚙️',
            'editorai': '📝',
            'ethosai': '⚖️',
            'searchai': '🔍'
        };
        agentIcon.textContent = icons[agentName.toLowerCase()] || '🤖';
    }

    async startSession() {
        try {
            this.updateStatus('connecting', 'Creating session...');
            this.showConnectionStatus('Creating your course session...');

            const courseData = {
                title: this.courseTitleInput.value,
                level: this.courseLevelInput.value,
                duration: parseInt(this.courseDurationInput.value),
                description: this.courseDescriptionInput.value
            };

            console.log('Creating session with data:', courseData);

            const response = await fetch(`${this.apiBaseUrl}/frontend/quick-start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(courseData)
            });

            if (!response.ok) {
                throw new Error(`Failed to create session: ${response.status} ${response.statusText}`);
            }

            const sessionData = await response.json();
            console.log('Session created:', sessionData);

            this.sessionId = sessionData.session_id;
            this.sessionDisplay.textContent = `Session: ${this.sessionId.substring(0, 8)}...`;

            // Switch to chat interface
            this.setupContainer.style.display = 'none';
            this.chatContainer.style.display = 'flex';

            // Connect WebSocket
            await this.connectWebSocket();

            // Send initial greeting
            this.addSystemMessage(`Course session created! I'm connecting you with the HAILEI team to design your "${courseData.title}" course.`);
            
            this.hideConnectionStatus();

        } catch (error) {
            console.error('Error starting session:', error);
            this.updateStatus('offline', 'Connection failed');
            this.hideConnectionStatus();
            alert(`Failed to start session: ${error.message}`);
        }
    }

    async connectWebSocket() {
        try {
            this.showConnectionStatus('Connecting to conversation...');
            
            const wsUrl = `ws://localhost:8000/ws/${this.sessionId}`;
            console.log('Connecting to WebSocket:', wsUrl);

            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateStatus('online', 'Connected');
                this.enableInput();
                this.showQuickActions();
                this.hideConnectionStatus();
            };

            this.websocket.onmessage = (event) => {
                console.log('WebSocket message received:', event.data);
                this.handleWebSocketMessage(event.data);
            };

            this.websocket.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                this.updateStatus('offline', 'Disconnected');
                this.disableInput();
                this.hideQuickActions();
                
                if (event.code !== 1000) { // Not normal closure
                    this.addSystemMessage('Connection lost. Please refresh the page to reconnect.');
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('offline', 'Connection error');
                this.addSystemMessage('Connection error occurred. Please check your connection and try again.');
            };

        } catch (error) {
            console.error('Error connecting WebSocket:', error);
            throw error;
        }
    }

    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            console.log('Parsed message:', message);

            switch (message.type) {
                case 'connection_established':
                    console.log('Connection established:', message.message);
                    this.addSystemMessage('Connected to HAILEI system');
                    break;
                    
                case 'conversation_started':
                    console.log('Conversation started:', message.message);
                    this.updateAgent(message.current_agent || 'coordinator', 'Starting conversation...');
                    this.addSystemMessage(message.message || 'Conversation started');
                    break;
                    
                case 'agent_message':
                    this.addAgentMessage(message.content, message.agent || this.currentAgent);
                    if (message.agent) {
                        this.updateAgentFromMessage(message.agent);
                    }
                    break;
                    
                case 'system_message':
                    this.addSystemMessage(message.content);
                    break;
                    
                case 'agent_transition':
                    this.updateAgent(message.agent, message.role || 'Processing');
                    this.addSystemMessage(`Now working with ${message.agent}: ${message.role || 'Processing your request'}`);
                    break;
                    
                case 'error':
                    this.addSystemMessage(`Error: ${message.content || message.message}`, 'error');
                    break;
                    
                case 'echo':
                    console.log('Message echoed:', message.original_message);
                    // Don't display echo messages to user
                    break;
                    
                case 'pong':
                    console.log('Pong received');
                    break;
                    
                case 'subscribed':
                    console.log('Subscribed to events:', message.events);
                    break;
                    
                default:
                    // Handle plain text messages
                    if (typeof message === 'string') {
                        this.addAgentMessage(message, this.currentAgent);
                    } else {
                        console.log('Unknown message type:', message);
                    }
            }
        } catch (error) {
            // Handle plain text messages
            this.addAgentMessage(data, this.currentAgent);
        }
    }

    updateAgentFromMessage(agentName) {
        const agentRoles = {
            'coordinator': 'Course Coordination',
            'ipdai': 'Instructional Design',
            'cauthai': 'Content Creation',
            'tfdai': 'Technical Implementation',
            'editorai': 'Content Review',
            'ethosai': 'Ethical Compliance',
            'searchai': 'Resource Research'
        };
        
        this.updateAgent(agentName, agentRoles[agentName.toLowerCase()] || 'Processing');
    }

    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || !this.isConnected) return;

        console.log('Sending message:', message);

        // Add user message to chat
        this.addUserMessage(message);

        // Send to WebSocket
        this.websocket.send(JSON.stringify({
            type: 'user_message',
            content: message,
            timestamp: new Date().toISOString()
        }));

        // Clear input
        this.messageInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
    }

    sendQuickMessage(message) {
        if (!this.isConnected) return;
        
        this.messageInput.value = message;
        this.sendMessage();
    }

    addUserMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.innerHTML = `
            <div class="message-bubble">
                <div class="message-content">${this.escapeHtml(content)}</div>
                <div class="message-time">${this.formatTime(new Date())}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addAgentMessage(content, agent = 'coordinator') {
        this.hideTypingIndicator();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message agent';
        
        const agentIcons = {
            'coordinator': '🎯',
            'ipdai': '📋',
            'cauthai': '✍️',
            'tfdai': '⚙️',
            'editorai': '📝',
            'ethosai': '⚖️',
            'searchai': '🔍'
        };
        
        const icon = agentIcons[agent.toLowerCase()] || '🤖';
        
        messageDiv.innerHTML = `
            <div class="agent-avatar">${icon}</div>
            <div class="message-content">
                <strong>${agent.toUpperCase()}</strong><br>
                ${this.escapeHtml(content)}
                <div class="message-time">${this.formatTime(new Date())}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addSystemMessage(content, type = 'info') {
        this.hideTypingIndicator();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message system ${type}`;
        
        const bgColor = type === 'error' ? '#fed7d7' : '#e6fffa';
        const borderColor = type === 'error' ? '#feb2b2' : '#81e6d9';
        
        messageDiv.innerHTML = `
            <div style="background: ${bgColor}; border: 1px solid ${borderColor}; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; width: 100%;">
                <div class="message-content">
                    <strong>System:</strong> ${this.escapeHtml(content)}
                    <div class="message-time">${this.formatTime(new Date())}</div>
                </div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.hideTypingIndicator(); // Remove existing one
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="agent-avatar">💭</div>
            <div>
                <strong>${this.currentAgent.toUpperCase()}</strong> is typing
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    enableInput() {
        this.messageInput.disabled = false;
        this.sendBtn.disabled = false;
        this.messageInput.placeholder = 'Type your message...';
        this.messageInput.focus();
    }

    disableInput() {
        this.messageInput.disabled = true;
        this.sendBtn.disabled = true;
        this.messageInput.placeholder = 'Disconnected...';
    }

    showQuickActions() {
        this.quickActions.style.display = 'flex';
    }

    hideQuickActions() {
        this.quickActions.style.display = 'none';
    }

    showConnectionStatus(message) {
        this.connectionMessage.textContent = message;
        this.connectionStatus.style.display = 'block';
    }

    hideConnectionStatus() {
        this.connectionStatus.style.display = 'none';
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
}

// Global function for quick actions
function sendQuickMessage(message) {
    if (window.haileiChat) {
        window.haileiChat.sendQuickMessage(message);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing HAILEI Chat...');
    window.haileiChat = new HAILEIChat();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.haileiChat && !window.haileiChat.isConnected) {
        console.log('Page became visible, checking connection...');
        // Could implement reconnection logic here
    }
});

// Handle beforeunload
window.addEventListener('beforeunload', () => {
    if (window.haileiChat && window.haileiChat.websocket) {
        window.haileiChat.websocket.close(1000, 'Page unload');
    }
});