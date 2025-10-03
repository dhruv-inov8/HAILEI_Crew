"""
HAILEI WebSocket Connection Manager

Real-time communication management for conversational AI orchestration.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """
    WebSocket connection manager for real-time communication.
    
    Features:
    - Session-based connection grouping
    - Broadcast messaging
    - Connection health monitoring
    - Automatic cleanup
    """
    
    def __init__(self):
        # Active connections by session
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Message queue for offline sessions
        self.message_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Statistics
        self.connection_stats = {
            'total_connections': 0,
            'active_sessions': 0,
            'messages_sent': 0,
            'last_activity': None
        }
        
        # Setup logging
        self.logger = logging.getLogger('websocket_manager')
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        # Add to active connections
        self.active_connections[session_id].add(websocket)
        
        # Store connection metadata
        self.connection_metadata[websocket] = {
            'session_id': session_id,
            'connected_at': datetime.now(),
            'last_ping': datetime.now(),
            'message_count': 0
        }
        
        # Update statistics
        self.connection_stats['total_connections'] += 1
        self.connection_stats['active_sessions'] = len(self.active_connections)
        self.connection_stats['last_activity'] = datetime.now()
        
        self.logger.info(f"WebSocket connected for session: {session_id}")
        
        # Send connection confirmation
        await self.send_to_connection(websocket, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to HAILEI real-time updates"
        })
        
        # Send any queued messages
        await self._send_queued_messages(websocket, session_id)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove WebSocket connection"""
        # Remove from active connections
        self.active_connections[session_id].discard(websocket)
        
        # Clean up empty sessions
        if not self.active_connections[session_id]:
            del self.active_connections[session_id]
        
        # Remove metadata
        self.connection_metadata.pop(websocket, None)
        
        # Update statistics
        self.connection_stats['active_sessions'] = len(self.active_connections)
        self.connection_stats['last_activity'] = datetime.now()
        
        self.logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific WebSocket connection"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()
            
            # Send message
            await websocket.send_text(json.dumps(message))
            
            # Update metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['message_count'] += 1
                self.connection_metadata[websocket]['last_ping'] = datetime.now()
            
            # Update statistics
            self.connection_stats['messages_sent'] += 1
            self.connection_stats['last_activity'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket message: {e}")
            # Connection is likely dead, will be cleaned up on next interaction
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send message to all connections in a session"""
        if session_id not in self.active_connections:
            # Queue message for when connections are available
            self.message_queue[session_id].append({
                **message,
                'queued_at': datetime.now().isoformat()
            })
            self.logger.info(f"Queued message for offline session: {session_id}")
            return
        
        # Send to all connections in session
        connections = list(self.active_connections[session_id])  # Copy to avoid modification during iteration
        
        for websocket in connections:
            try:
                await self.send_to_connection(websocket, message)
            except Exception as e:
                self.logger.error(f"Failed to send to connection in session {session_id}: {e}")
                # Remove failed connection
                self.disconnect(websocket, session_id)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        for session_id in list(self.active_connections.keys()):
            await self.send_to_session(session_id, message)
    
    async def send_agent_update(self, session_id: str, agent_id: str, update_type: str, data: Dict[str, Any]):
        """Send agent-specific update"""
        message = {
            "type": "agent_update",
            "agent_id": agent_id,
            "update_type": update_type,
            "data": data,
            "session_id": session_id
        }
        
        await self.send_to_session(session_id, message)
    
    async def send_phase_update(self, session_id: str, phase_id: str, update_type: str, data: Dict[str, Any]):
        """Send phase-specific update"""
        message = {
            "type": "phase_update",
            "phase_id": phase_id,
            "update_type": update_type,
            "data": data,
            "session_id": session_id
        }
        
        await self.send_to_session(session_id, message)
    
    async def send_error_notification(self, session_id: str, error_type: str, error_message: str, details: Dict[str, Any] = None):
        """Send error notification"""
        message = {
            "type": "error_notification",
            "error_type": error_type,
            "error_message": error_message,
            "details": details or {},
            "session_id": session_id
        }
        
        await self.send_to_session(session_id, message)
    
    async def send_system_notification(self, message_text: str, level: str = "info"):
        """Send system-wide notification"""
        message = {
            "type": "system_notification",
            "message": message_text,
            "level": level,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast_to_all(message)
    
    async def ping_connections(self):
        """Send ping to all connections to check health"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect failed connections
        failed_connections = []
        
        for session_id, connections in self.active_connections.items():
            for websocket in list(connections):
                try:
                    await self.send_to_connection(websocket, ping_message)
                except Exception:
                    failed_connections.append((websocket, session_id))
        
        # Clean up failed connections
        for websocket, session_id in failed_connections:
            self.disconnect(websocket, session_id)
        
        if failed_connections:
            self.logger.info(f"Cleaned up {len(failed_connections)} failed connections")
    
    async def _send_queued_messages(self, websocket: WebSocket, session_id: str):
        """Send queued messages to newly connected client"""
        if session_id in self.message_queue:
            queued_messages = self.message_queue[session_id]
            
            # Send each queued message
            for message in queued_messages:
                try:
                    await self.send_to_connection(websocket, {
                        **message,
                        "queued": True
                    })
                except Exception as e:
                    self.logger.error(f"Failed to send queued message: {e}")
                    break
            
            # Clear the queue
            self.message_queue[session_id].clear()
            self.logger.info(f"Sent {len(queued_messages)} queued messages to session: {session_id}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_session_connection_count(self, session_id: str) -> int:
        """Get number of connections for specific session"""
        return len(self.active_connections.get(session_id, set()))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self.connection_stats,
            'current_connections': self.get_connection_count(),
            'sessions_with_connections': len(self.active_connections),
            'queued_messages': sum(len(queue) for queue in self.message_queue.values())
        }
    
    def get_session_list(self) -> List[str]:
        """Get list of sessions with active connections"""
        return list(self.active_connections.keys())
    
    def is_session_connected(self, session_id: str) -> bool:
        """Check if session has any active connections"""
        return session_id in self.active_connections and len(self.active_connections[session_id]) > 0
    
    async def cleanup_stale_connections(self, max_idle_minutes: int = 30):
        """Clean up connections that have been idle too long"""
        cutoff_time = datetime.now().timestamp() - (max_idle_minutes * 60)
        stale_connections = []
        
        for websocket, metadata in self.connection_metadata.items():
            if metadata['last_ping'].timestamp() < cutoff_time:
                stale_connections.append((websocket, metadata['session_id']))
        
        # Disconnect stale connections
        for websocket, session_id in stale_connections:
            try:
                await websocket.close(code=1000, reason="Connection idle timeout")
            except Exception:
                pass
            self.disconnect(websocket, session_id)
        
        if stale_connections:
            self.logger.info(f"Cleaned up {len(stale_connections)} stale connections")
    
    async def send_progress_update(self, session_id: str, operation: str, progress: float, details: Dict[str, Any] = None):
        """Send progress update for long-running operations"""
        message = {
            "type": "progress_update",
            "operation": operation,
            "progress": min(100.0, max(0.0, progress)),  # Clamp between 0-100
            "details": details or {},
            "session_id": session_id
        }
        
        await self.send_to_session(session_id, message)
    
    async def send_conversation_update(self, session_id: str, speaker: str, content: str, metadata: Dict[str, Any] = None):
        """Send conversation message update"""
        message = {
            "type": "conversation_update",
            "speaker": speaker,
            "content": content,
            "metadata": metadata or {},
            "session_id": session_id
        }
        
        await self.send_to_session(session_id, message)


# Background task for connection health monitoring
async def connection_health_monitor(connection_manager: ConnectionManager, interval_seconds: int = 30):
    """Background task to monitor connection health"""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            
            # Ping all connections
            await connection_manager.ping_connections()
            
            # Clean up stale connections
            await connection_manager.cleanup_stale_connections()
            
        except Exception as e:
            logging.error(f"Connection health monitor error: {e}")
            await asyncio.sleep(5)  # Brief pause before retrying