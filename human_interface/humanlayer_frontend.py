"""
HAILEI Frontend HumanLayer Integration

Production-ready human interaction system designed for Flask/FastAPI deployment
with WebSocket support and conversational UI components.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from humanlayer import HumanLayer


class InteractionType(Enum):
    """Types of human interactions for frontend handling"""
    APPROVAL = "approval"
    FEEDBACK = "feedback"
    CHOICE = "choice"
    TEXT_INPUT = "text_input"
    REVIEW = "review"
    REFINEMENT = "refinement"


@dataclass
class InteractionRequest:
    """Structured interaction request for frontend display"""
    request_id: str
    session_id: str
    interaction_type: InteractionType
    title: str
    message: str
    options: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    required: bool = True
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        data = asdict(self)
        data['interaction_type'] = self.interaction_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionRequest':
        """Create from dictionary for API deserialization"""
        data['interaction_type'] = InteractionType(data['interaction_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class InteractionResponse:
    """Structured interaction response from user"""
    request_id: str
    session_id: str
    response_type: str
    response_data: Any
    timestamp: datetime = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'request_id': self.request_id,
            'session_id': self.session_id,
            'response_type': self.response_type,
            'response_data': self.response_data,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id
        }


class FrontendHumanLayer:
    """
    Production-ready HumanLayer integration for frontend deployment.
    
    Features:
    - WebSocket-compatible real-time interactions
    - Flask/FastAPI endpoint integration
    - Timeout handling and retry mechanisms
    - Response validation and error handling
    - Session-based interaction management
    - UI component data formatting
    """
    
    def __init__(
        self,
        humanlayer_instance: Optional[HumanLayer] = None,
        default_timeout: int = 300,  # 5 minutes
        enable_websocket: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize frontend HumanLayer integration.
        
        Args:
            humanlayer_instance: HumanLayer instance for backend integration
            default_timeout: Default timeout in seconds for user responses
            enable_websocket: Enable WebSocket support for real-time updates
            enable_logging: Enable detailed logging
        """
        self.humanlayer = humanlayer_instance or HumanLayer()
        self.default_timeout = default_timeout
        self.enable_websocket = enable_websocket
        
        # Interaction management
        self.pending_interactions: Dict[str, InteractionRequest] = {}
        self.interaction_responses: Dict[str, InteractionResponse] = {}
        self.interaction_callbacks: Dict[str, Callable] = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, Any] = {}  # session_id -> websocket
        
        # Event callbacks for frontend integration
        self.event_callbacks: Dict[str, List[Callable]] = {
            'interaction_requested': [],
            'interaction_completed': [],
            'interaction_timeout': [],
            'session_connected': [],
            'session_disconnected': []
        }
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger('hailei_humanlayer_frontend')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logging.getLogger('null')
            self.logger.addHandler(logging.NullHandler())
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for frontend event handling"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks and WebSocket connections"""
        # Call registered callbacks
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Event callback error for {event_type}: {e}")
        
        # Send to WebSocket connections if enabled
        if self.enable_websocket and 'session_id' in data:
            session_id = data['session_id']
            if session_id in self.websocket_connections:
                asyncio.create_task(self._send_websocket_message(session_id, {
                    'event_type': event_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }))
    
    async def _send_websocket_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to WebSocket connection"""
        try:
            websocket = self.websocket_connections.get(session_id)
            if websocket:
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"WebSocket send error for session {session_id}: {e}")
    
    def connect_websocket(self, session_id: str, websocket):
        """Connect WebSocket for session"""
        self.websocket_connections[session_id] = websocket
        self.emit_event('session_connected', {'session_id': session_id})
        self.logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect_websocket(self, session_id: str):
        """Disconnect WebSocket for session"""
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
            self.emit_event('session_disconnected', {'session_id': session_id})
            self.logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def request_approval(
        self,
        session_id: str,
        title: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Request approval from user with frontend integration.
        
        Args:
            session_id: Session identifier
            title: Approval request title
            message: Approval message
            context: Additional context for frontend display
            timeout: Timeout in seconds
            
        Returns:
            bool: True if approved, False if denied
        """
        request = InteractionRequest(
            request_id=f"approval_{datetime.now().timestamp()}",
            session_id=session_id,
            interaction_type=InteractionType.APPROVAL,
            title=title,
            message=message,
            options=["approve", "deny"],
            context=context,
            timeout_seconds=timeout or self.default_timeout
        )
        
        response = await self._handle_interaction_request(request)
        
        # Parse approval response
        if isinstance(response.response_data, str):
            return response.response_data.lower() in ['approve', 'approved', 'yes', 'true']
        elif isinstance(response.response_data, bool):
            return response.response_data
        else:
            return False
    
    async def request_feedback(
        self,
        session_id: str,
        title: str,
        message: str,
        current_content: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Request feedback from user for content refinement.
        
        Args:
            session_id: Session identifier
            title: Feedback request title
            message: Feedback message
            current_content: Current content for review
            context: Additional context for frontend display
            timeout: Timeout in seconds
            
        Returns:
            str: User feedback text
        """
        request = InteractionRequest(
            request_id=f"feedback_{datetime.now().timestamp()}",
            session_id=session_id,
            interaction_type=InteractionType.FEEDBACK,
            title=title,
            message=message,
            context={
                'current_content': current_content,
                **(context or {})
            },
            timeout_seconds=timeout or self.default_timeout
        )
        
        response = await self._handle_interaction_request(request)
        return str(response.response_data)
    
    async def request_choice(
        self,
        session_id: str,
        title: str,
        message: str,
        options: List[str],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Request choice selection from user.
        
        Args:
            session_id: Session identifier
            title: Choice request title
            message: Choice message
            options: Available options
            context: Additional context for frontend display
            timeout: Timeout in seconds
            
        Returns:
            str: Selected option
        """
        request = InteractionRequest(
            request_id=f"choice_{datetime.now().timestamp()}",
            session_id=session_id,
            interaction_type=InteractionType.CHOICE,
            title=title,
            message=message,
            options=options,
            context=context,
            timeout_seconds=timeout or self.default_timeout
        )
        
        response = await self._handle_interaction_request(request)
        return str(response.response_data)
    
    async def request_text_input(
        self,
        session_id: str,
        title: str,
        message: str,
        placeholder: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Request text input from user.
        
        Args:
            session_id: Session identifier
            title: Input request title
            message: Input message
            placeholder: Placeholder text for input field
            context: Additional context for frontend display
            timeout: Timeout in seconds
            
        Returns:
            str: User input text
        """
        request = InteractionRequest(
            request_id=f"text_{datetime.now().timestamp()}",
            session_id=session_id,
            interaction_type=InteractionType.TEXT_INPUT,
            title=title,
            message=message,
            context={
                'placeholder': placeholder,
                **(context or {})
            },
            timeout_seconds=timeout or self.default_timeout
        )
        
        response = await self._handle_interaction_request(request)
        return str(response.response_data)
    
    async def request_content_review(
        self,
        session_id: str,
        title: str,
        content: str,
        agent_name: str,
        phase_name: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Request comprehensive content review from user.
        
        Args:
            session_id: Session identifier
            title: Review request title
            content: Content to review
            agent_name: Name of agent that generated content
            phase_name: Current phase name
            context: Additional context for frontend display
            timeout: Timeout in seconds
            
        Returns:
            Dict with review decision and optional feedback
        """
        request = InteractionRequest(
            request_id=f"review_{datetime.now().timestamp()}",
            session_id=session_id,
            interaction_type=InteractionType.REVIEW,
            title=title,
            message=f"Please review the output from {agent_name} for {phase_name}",
            context={
                'content': content,
                'agent_name': agent_name,
                'phase_name': phase_name,
                'review_options': ['approve', 'request_changes', 'reject'],
                **(context or {})
            },
            timeout_seconds=timeout or self.default_timeout
        )
        
        response = await self._handle_interaction_request(request)
        
        # Parse review response
        if isinstance(response.response_data, dict):
            return response.response_data
        else:
            # Handle simple string responses
            return {
                'decision': str(response.response_data),
                'feedback': ''
            }
    
    async def _handle_interaction_request(self, request: InteractionRequest) -> InteractionResponse:
        """
        Handle interaction request with timeout and error management.
        
        Args:
            request: Interaction request
            
        Returns:
            InteractionResponse: User response
        """
        self.logger.info(f"Requesting interaction: {request.interaction_type.value} for session {request.session_id}")
        
        # Store pending interaction
        self.pending_interactions[request.request_id] = request
        
        # Emit interaction requested event
        self.emit_event('interaction_requested', {
            'session_id': request.session_id,
            'request': request.to_dict()
        })
        
        try:
            # Wait for response with timeout
            response = await self._wait_for_response(request)
            
            # Clean up pending interaction
            if request.request_id in self.pending_interactions:
                del self.pending_interactions[request.request_id]
            
            # Emit interaction completed event
            self.emit_event('interaction_completed', {
                'session_id': request.session_id,
                'request_id': request.request_id,
                'response': response.to_dict()
            })
            
            self.logger.info(f"Interaction completed: {request.request_id}")
            return response
            
        except asyncio.TimeoutError:
            # Handle timeout
            self.logger.warning(f"Interaction timeout: {request.request_id}")
            
            # Clean up pending interaction
            if request.request_id in self.pending_interactions:
                del self.pending_interactions[request.request_id]
            
            # Emit timeout event
            self.emit_event('interaction_timeout', {
                'session_id': request.session_id,
                'request_id': request.request_id
            })
            
            # Return default response based on interaction type
            return self._get_default_response(request)
        
        except Exception as e:
            self.logger.error(f"Interaction error for {request.request_id}: {e}")
            
            # Clean up pending interaction
            if request.request_id in self.pending_interactions:
                del self.pending_interactions[request.request_id]
            
            raise
    
    async def _wait_for_response(self, request: InteractionRequest) -> InteractionResponse:
        """Wait for user response with timeout"""
        timeout_seconds = request.timeout_seconds or self.default_timeout
        end_time = datetime.now() + timedelta(seconds=timeout_seconds)
        
        while datetime.now() < end_time:
            # Check if response received
            if request.request_id in self.interaction_responses:
                response = self.interaction_responses[request.request_id]
                del self.interaction_responses[request.request_id]
                return response
            
            # Wait briefly before checking again
            await asyncio.sleep(0.1)
        
        # Timeout reached
        raise asyncio.TimeoutError(f"No response received for {request.request_id}")
    
    def _get_default_response(self, request: InteractionRequest) -> InteractionResponse:
        """Get default response for timeout scenarios"""
        default_data = {
            InteractionType.APPROVAL: False,
            InteractionType.FEEDBACK: "No feedback provided (timeout)",
            InteractionType.CHOICE: request.options[0] if request.options else "default",
            InteractionType.TEXT_INPUT: "",
            InteractionType.REVIEW: {"decision": "timeout", "feedback": ""},
            InteractionType.REFINEMENT: {"decision": "timeout", "feedback": ""}
        }
        
        return InteractionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            response_type="timeout",
            response_data=default_data.get(request.interaction_type, "timeout")
        )
    
    def submit_response(
        self,
        request_id: str,
        response_data: Any,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Submit user response to pending interaction.
        
        Args:
            request_id: Request identifier
            response_data: User response data
            user_id: Optional user identifier
            
        Returns:
            bool: Success status
        """
        if request_id not in self.pending_interactions:
            self.logger.warning(f"No pending interaction found for request: {request_id}")
            return False
        
        request = self.pending_interactions[request_id]
        
        # Validate response data
        if not self._validate_response(request, response_data):
            self.logger.warning(f"Invalid response data for request: {request_id}")
            return False
        
        # Create response
        response = InteractionResponse(
            request_id=request_id,
            session_id=request.session_id,
            response_type=request.interaction_type.value,
            response_data=response_data,
            user_id=user_id
        )
        
        # Store response
        self.interaction_responses[request_id] = response
        
        self.logger.info(f"Response submitted for request: {request_id}")
        return True
    
    def _validate_response(self, request: InteractionRequest, response_data: Any) -> bool:
        """Validate response data against request requirements"""
        if request.interaction_type == InteractionType.APPROVAL:
            return isinstance(response_data, (bool, str))
        
        elif request.interaction_type == InteractionType.CHOICE:
            if isinstance(response_data, str) and request.options:
                return response_data in request.options
            return True  # Allow any string for flexibility
        
        elif request.interaction_type == InteractionType.FEEDBACK:
            return isinstance(response_data, str)
        
        elif request.interaction_type == InteractionType.TEXT_INPUT:
            return isinstance(response_data, str)
        
        elif request.interaction_type == InteractionType.REVIEW:
            if isinstance(response_data, dict):
                return 'decision' in response_data
            return isinstance(response_data, str)
        
        # Default: accept any response
        return True
    
    def get_pending_interactions(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending interactions for session or all sessions"""
        if session_id:
            return [
                request.to_dict() 
                for request in self.pending_interactions.values()
                if request.session_id == session_id
            ]
        else:
            return [request.to_dict() for request in self.pending_interactions.values()]
    
    def cancel_interaction(self, request_id: str) -> bool:
        """Cancel pending interaction"""
        if request_id in self.pending_interactions:
            request = self.pending_interactions[request_id]
            del self.pending_interactions[request_id]
            
            # Emit cancellation event
            self.emit_event('interaction_cancelled', {
                'session_id': request.session_id,
                'request_id': request_id
            })
            
            self.logger.info(f"Interaction cancelled: {request_id}")
            return True
        
        return False
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current state for session"""
        pending = [
            request.to_dict() 
            for request in self.pending_interactions.values()
            if request.session_id == session_id
        ]
        
        return {
            'session_id': session_id,
            'pending_interactions': pending,
            'websocket_connected': session_id in self.websocket_connections,
            'timestamp': datetime.now().isoformat()
        }