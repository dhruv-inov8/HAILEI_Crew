"""
HAILEI Frontend WebSocket Handlers

Enhanced WebSocket functionality specifically designed for frontend integration
with real-time updates, event streaming, and interactive communication.
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from .websocket_manager import ConnectionManager

logger = logging.getLogger("frontend_websocket")

# Frontend WebSocket event models
class WebSocketEvent(BaseModel):
    """Base WebSocket event model"""
    type: str
    timestamp: datetime = datetime.now()


class ClientMessage(BaseModel):
    """Message from client to server"""
    type: str
    data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class ServerMessage(BaseModel):
    """Message from server to client"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()


class ProgressUpdate(BaseModel):
    """Progress update event"""
    agent_id: str
    status: str
    progress: float
    message: str
    phase: str


class ChatEvent(BaseModel):
    """Chat interaction event"""
    message: str
    sender: str  # user, agent, system
    agent_id: Optional[str] = None
    timestamp: datetime = datetime.now()


class FrontendWebSocketManager:
    """Enhanced WebSocket manager for frontend integration"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.event_handlers = {
            "ping": self._handle_ping,
            "subscribe": self._handle_subscribe,
            "chat": self._handle_chat,
            "action": self._handle_action,
            "status_request": self._handle_status_request
        }
    
    async def handle_client_connection(self, websocket: WebSocket, session_id: str):
        """Handle frontend WebSocket connection with enhanced features"""
        await self.connection_manager.connect(websocket, session_id)
        
        # Send welcome message
        await self._send_to_client(websocket, {
            "type": "connected",
            "session_id": session_id,
            "features": [
                "real_time_progress",
                "chat_interface", 
                "agent_status_updates",
                "interactive_approvals"
            ],
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                await self._process_client_message(websocket, session_id, data)
                
        except WebSocketDisconnect:
            self.connection_manager.disconnect(websocket, session_id)
            logger.info(f"Frontend WebSocket disconnected for session: {session_id}")
        except Exception as e:
            logger.error(f"Frontend WebSocket error for session {session_id}: {e}")
            self.connection_manager.disconnect(websocket, session_id)
    
    async def _process_client_message(self, websocket: WebSocket, session_id: str, data: str):
        """Process incoming message from client"""
        try:
            message_data = json.loads(data)
            message = ClientMessage(**message_data)
            
            # Route to appropriate handler
            handler = self.event_handlers.get(message.type, self._handle_unknown)
            await handler(websocket, session_id, message)
            
        except ValidationError as e:
            await self._send_error(websocket, "Invalid message format", str(e))
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON", "Could not parse message")
        except Exception as e:
            logger.error(f"Error processing client message: {e}")
            await self._send_error(websocket, "Processing error", str(e))
    
    async def _handle_ping(self, websocket: WebSocket, session_id: str, message: ClientMessage):
        """Handle ping messages"""
        await self._send_to_client(websocket, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_subscribe(self, websocket: WebSocket, session_id: str, message: ClientMessage):
        """Handle subscription to specific events"""
        events = message.data.get("events", []) if message.data else []
        
        # Store subscription preferences (could be persisted)
        await self._send_to_client(websocket, {
            "type": "subscribed",
            "events": events,
            "message": f"Subscribed to {len(events)} event types",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_chat(self, websocket: WebSocket, session_id: str, message: ClientMessage):
        """Handle chat messages from frontend"""
        if not message.data or "message" not in message.data:
            await self._send_error(websocket, "Invalid chat message", "Message content required")
            return
        
        chat_text = message.data["message"]
        
        # Echo the message (in real implementation, would process through orchestrator)
        await self.broadcast_chat_event(session_id, ChatEvent(
            message=f"Processing: {chat_text}",
            sender="system",
            agent_id="orchestrator"
        ))
        
        # Simulate processing delay
        await asyncio.sleep(1)
        
        # Send response
        await self.broadcast_chat_event(session_id, ChatEvent(
            message=f"Response to: {chat_text}",
            sender="agent",
            agent_id="ipdai_agent"
        ))
    
    async def _handle_action(self, websocket: WebSocket, session_id: str, message: ClientMessage):
        """Handle action requests from frontend"""
        if not message.data or "action" not in message.data:
            await self._send_error(websocket, "Invalid action", "Action type required")
            return
        
        action_type = message.data["action"]
        
        # Process the action (would integrate with orchestrator)
        await self._send_to_client(websocket, {
            "type": "action_received",
            "action": action_type,
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        })
        
        # Simulate action processing
        await asyncio.sleep(2)
        
        await self.broadcast_to_session(session_id, {
            "type": "action_completed",
            "action": action_type,
            "status": "completed",
            "result": f"Action '{action_type}' completed successfully",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_status_request(self, websocket: WebSocket, session_id: str, message: ClientMessage):
        """Handle status requests"""
        # Mock status response
        status = {
            "session_id": session_id,
            "current_phase": "foundation_design",
            "progress": 0.75,
            "active_agents": ["ipdai_agent", "cauthai_agent"],
            "last_update": datetime.now().isoformat()
        }
        
        await self._send_to_client(websocket, {
            "type": "status_response",
            "data": status,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_unknown(self, websocket: WebSocket, session_id: str, message: ClientMessage):
        """Handle unknown message types"""
        await self._send_error(websocket, "Unknown message type", f"Type '{message.type}' not supported")
    
    async def _send_to_client(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")
    
    async def _send_error(self, websocket: WebSocket, error: str, details: str):
        """Send error message to client"""
        await self._send_to_client(websocket, {
            "type": "error",
            "error": error,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    # Public methods for broadcasting events
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a session"""
        await self.connection_manager.send_to_session(session_id, message)
    
    async def broadcast_progress_update(self, session_id: str, progress: ProgressUpdate):
        """Broadcast progress update to session"""
        await self.broadcast_to_session(session_id, {
            "type": "progress_update",
            "data": progress.model_dump(),
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_chat_event(self, session_id: str, chat: ChatEvent):
        """Broadcast chat event to session"""
        await self.broadcast_to_session(session_id, {
            "type": "chat_event",
            "data": chat.model_dump(),
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_phase_change(self, session_id: str, from_phase: str, to_phase: str):
        """Broadcast phase transition to session"""
        await self.broadcast_to_session(session_id, {
            "type": "phase_change",
            "data": {
                "from_phase": from_phase,
                "to_phase": to_phase,
                "message": f"Transitioning from {from_phase} to {to_phase}"
            },
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_agent_status(self, session_id: str, agent_id: str, status: str, message: str):
        """Broadcast agent status update to session"""
        await self.broadcast_to_session(session_id, {
            "type": "agent_status",
            "data": {
                "agent_id": agent_id,
                "status": status,
                "message": message
            },
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_system_notification(self, session_id: str, level: str, message: str):
        """Broadcast system notification to session"""
        await self.broadcast_to_session(session_id, {
            "type": "system_notification",
            "data": {
                "level": level,  # info, warning, error, success
                "message": message
            },
            "timestamp": datetime.now().isoformat()
        })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for monitoring"""
        return self.connection_manager.get_connection_stats()


# Global frontend WebSocket manager instance
frontend_ws_manager = FrontendWebSocketManager()