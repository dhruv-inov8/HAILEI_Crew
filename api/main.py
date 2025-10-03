"""
HAILEI FastAPI Main Application

Production-ready FastAPI backend with WebSocket support for real-time
conversational AI orchestration with frontend integration.
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from orchestrator import (
    HAILEIOrchestrator, 
    ConversationState, 
    ContextType, 
    MemoryPriority,
    ErrorSeverity
)
from agents.conversational_agents import ConversationalAgentFactory
from hailei_crew_setup import HAILEICourseDesign
from .models import *
from .websocket_manager import ConnectionManager
from .auth import verify_token, get_current_user, auth_router
from .middleware import setup_middleware


# Global orchestrator instance
orchestrator: Optional[HAILEIOrchestrator] = None
connection_manager = ConnectionManager()

# Session logging system
session_logs = {}

class SessionLogger:
    """Captures all terminal output for a specific session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_file = f"session_logs/session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create logs directory if it doesn't exist
        os.makedirs("session_logs", exist_ok=True)
        
        # Create log file and write header
        with open(self.log_file, 'w') as f:
            f.write(f"=== HAILEI Session Log ===\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write(f"{'='*50}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message to the session file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    
    def log_websocket_message(self, direction: str, message: dict):
        """Log WebSocket messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] WebSocket {direction}: {json.dumps(message, indent=2)}\n")
    
    def log_agent_output(self, agent_id: str, output: str):
        """Log agent outputs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] AGENT OUTPUT ({agent_id}):\n")
            f.write(f"{'='*60}\n")
            f.write(f"{output}\n")
            f.write(f"{'='*60}\n\n")
    
    def close(self):
        """Close the session log"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Session End Time: {datetime.now().isoformat()}\n")
            f.write(f"=== End Session Log ===\n")

def get_session_logger(session_id: str) -> SessionLogger:
    """Get or create session logger"""
    if session_id not in session_logs:
        session_logs[session_id] = SessionLogger(session_id)
    return session_logs[session_id]

async def detect_message_intent(user_message: str) -> str:
    """
    Use LLM to intelligently detect user message intent for approval/feedback detection.
    
    Returns:
        str: "approval", "feedback", or "chat"
    """
    if not user_message.strip():
        return "chat"
    
    try:
        # Use a lightweight LiteLLM call for intent detection (compatible with existing setup)
        from litellm import acompletion
        
        system_prompt = """You are a message intent classifier for an educational AI system. 
Users are reviewing course content created by AI agents and can either:
1. APPROVE the content (proceed to next phase)
2. REQUEST FEEDBACK/CHANGES (modify the content) 
3. ASK QUESTIONS or CHAT (general conversation)

Classify the user's message intent. Respond with ONLY ONE WORD:
- "approval" - if user is approving, confirming, accepting, proceeding, or showing satisfaction
- "feedback" - if user wants changes, modifications, improvements, or is expressing dissatisfaction  
- "chat" - if user is asking questions, chatting, or unclear intent

Examples:
"I approve" → approval
"Looks good" → approval  
"Continue" → approval
"I confirm this" → approval
"Let's proceed" → approval
"Yes, this works" → approval
"Perfect!" → approval
"Can you change X" → feedback
"Make it shorter" → feedback
"I don't like Y" → feedback
"Modify the structure" → feedback
"What does KDKA mean?" → chat
"How does this work?" → chat
"Tell me more about..." → chat"""

        response = await acompletion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        intent = response.choices[0].message.content.strip().lower()
        
        # Validate response and default to chat if unclear
        if intent in ["approval", "feedback", "chat"]:
            return intent
        else:
            logging.warning(f"Unexpected intent classification: {intent}, defaulting to chat")
            return "chat"
            
    except Exception as e:
        logging.error(f"Failed to classify message intent: {e}")
        # Fallback to simple keyword detection if LLM fails
        content_lower = user_message.lower()
        if any(word in content_lower for word in ["approve", "confirm", "yes", "good", "continue", "proceed"]):
            return "approval"
        elif any(word in content_lower for word in ["change", "modify", "different", "improve", "feedback", "no"]):
            return "feedback"
        else:
            return "chat"

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestrator
    
    # Startup
    logging.info("🚀 Starting HAILEI API server...")
    
    # Initialize orchestrator
    try:
        # Load frameworks
        frameworks = {
            'kdka': {
                'name': 'Knowledge, Delivery, Context, Assessment',
                'summary': 'Comprehensive instructional design framework',
                'components': ['Knowledge Analysis', 'Delivery Methods', 'Context Adaptation', 'Assessment Design']
            },
            'prrr': {
                'name': 'Personal, Relatable, Relative, Real-world',
                'summary': 'Engagement framework for learner connection',
                'components': ['Personal Connection', 'Relatable Examples', 'Relative Context', 'Real-world Application']
            }
        }
        
        # Initialize HAILEI crew system to get agents
        logging.info("🔧 Initializing HAILEI crew system...")
        hailei_crew = HAILEICourseDesign()
        
        # Extract agents from crew system
        crew_agents = {
            'hailei4t_coordinator_agent': hailei_crew.hailei4t_coordinator_agent(),
            'ipdai_agent': hailei_crew.ipdai_agent(),
            'cauthai_agent': hailei_crew.cauthai_agent(),
            'tfdai_agent': hailei_crew.tfdai_agent(),
            'editorai_agent': hailei_crew.editorai_agent(),
            'ethosai_agent': hailei_crew.ethosai_agent(),
            'searchai_agent': hailei_crew.searchai_agent()
        }
        
        logging.info(f"✅ Loaded {len(crew_agents)} CrewAI agents")
        
        # Initialize orchestrator with actual agents
        orchestrator = HAILEIOrchestrator(
            agents=crew_agents,
            frameworks=frameworks,
            max_parallel_agents=5,
            enable_logging=True
        )
        
        # Store global frameworks context
        orchestrator.store_global_context(
            content={
                'frameworks': frameworks,
                'system_info': {
                    'name': 'HAILEI',
                    'version': '1.0.0',
                    'description': 'Conversational AI Orchestration System',
                    'startup_time': datetime.now().isoformat()
                }
            },
            priority=MemoryPriority.CRITICAL,
            tags={'system', 'frameworks', 'startup'}
        )
        
        logging.info("✅ HAILEI Orchestrator initialized successfully")
        
        # Inject orchestrator into frontend endpoints
        try:
            set_orchestrator(orchestrator)
            logging.info("✅ Orchestrator injected into frontend endpoints")
        except Exception as e:
            logging.error(f"❌ Failed to inject orchestrator into frontend: {e}")
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize orchestrator: {e}")
        # Log the full traceback for debugging
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        
        # Set orchestrator to None so health checks will indicate degraded state
        orchestrator = None
        logging.warning("🔧 System running in degraded mode - orchestrator unavailable")
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down HAILEI API server...")
    if orchestrator:
        orchestrator.shutdown()


# FastAPI app with lifespan management
app = FastAPI(
    title="HAILEI Conversational AI API",
    description="Production-ready API for educational AI course design with conversational workflows",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Include routers
app.include_router(auth_router)

# Import and include frontend router after app initialization
try:
    from .frontend_endpoints import frontend_router, set_orchestrator
    app.include_router(frontend_router)
    logging.info("✅ Frontend router included successfully")
except Exception as e:
    logging.error(f"❌ Failed to include frontend router: {e}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "HAILEI Conversational AI Orchestration API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "websocket": "/ws/{session_id}"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check"""
    global orchestrator
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        if orchestrator:
            # Check orchestrator components
            health_status["components"] = {
                "orchestrator": "healthy",
                "active_sessions": len(orchestrator.active_sessions),
                "parallel_orchestrator": "healthy",
                "decision_engine": "healthy",
                "context_manager": "healthy",
                "error_recovery": "healthy"
            }
            
            # Get system statistics
            stats = {
                "parallel_execution": orchestrator.get_parallel_execution_statistics(),
                "decision_engine": orchestrator.get_decision_engine_statistics(),
                "context_manager": orchestrator.get_context_manager_statistics(),
                "error_recovery": orchestrator.get_error_recovery_statistics()
            }
            
            health_status["statistics"] = stats
        else:
            health_status["status"] = "degraded"
            health_status["components"]["orchestrator"] = "not_initialized"
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


@app.post("/sessions", response_model=SessionResponse, tags=["Sessions"])
async def create_session(
    request: CreateSessionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create new conversational session"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    try:
        # Create conversation session
        session_id = await orchestrator.start_conversation(
            course_request=request.course_request.model_dump(),
            session_id=request.session_id
        )
        
        # Initialize session logger
        session_logger = get_session_logger(session_id)
        session_logger.log(f"Session created: {session_id}")
        session_logger.log(f"Course request: {request.course_request.model_dump()}")
        
        # Store user preferences if provided
        if request.user_preferences:
            orchestrator.store_global_context(
                content={
                    'user_preferences': request.user_preferences.model_dump(),
                    'user_id': current_user.get('user_id'),
                    'session_id': session_id
                },
                priority=MemoryPriority.HIGH,
                tags={'user_preferences', 'session', session_id}
            )
        
        # Schedule background session setup
        background_tasks.add_task(
            setup_session_background,
            session_id,
            request.course_request.model_dump()
        )
        
        return SessionResponse(
            session_id=session_id,
            status="created",
            message="Session created successfully",
            created_at=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get("/sessions/{session_id}", response_model=SessionStateResponse, tags=["Sessions"])
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get session state and progress"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    session_state = orchestrator.get_session_state(session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStateResponse(
        session_id=session_id,
        session_state=session_state,
        timestamp=datetime.now()
    )


@app.post("/sessions/{session_id}/agents/{agent_id}/execute", response_model=AgentExecutionResponse, tags=["Agents"])
async def execute_agent(
    session_id: str,
    agent_id: str,
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute specific agent with task"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    # Load session
    conversation_state = orchestrator.load_session(session_id)
    if not conversation_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Execute agent
        agent_output = await orchestrator.activate_agent(
            agent_id=agent_id,
            task_description=request.task_description,
            context=request.context or {}
        )
        
        # Notify WebSocket clients
        background_tasks.add_task(
            notify_websocket_clients,
            session_id,
            {
                "type": "agent_completed",
                "agent_id": agent_id,
                "output": agent_output.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return AgentExecutionResponse(
            agent_id=agent_id,
            output=agent_output.to_dict(),
            execution_time=1.0,  # Placeholder
            success=True,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Agent execution failed: {e}")
        
        # Notify WebSocket clients of error
        background_tasks.add_task(
            notify_websocket_clients,
            session_id,
            {
                "type": "agent_error",
                "agent_id": agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.post("/sessions/{session_id}/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    session_id: str,
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Submit user feedback for agent refinement"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    conversation_state = orchestrator.load_session(session_id)
    if not conversation_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Process feedback
        refined_output = await orchestrator.process_user_feedback(
            agent_id=request.agent_id,
            feedback=request.feedback,
            phase_id=request.phase_id
        )
        
        # Notify WebSocket clients
        background_tasks.add_task(
            notify_websocket_clients,
            session_id,
            {
                "type": "feedback_processed",
                "agent_id": request.agent_id,
                "feedback": request.feedback,
                "refined_output": refined_output.to_dict() if refined_output else None,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return FeedbackResponse(
            feedback_id=f"fb_{session_id}_{int(datetime.now().timestamp())}",
            status="processed",
            refined_output=refined_output.to_dict() if refined_output else None,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")


@app.post("/sessions/{session_id}/approve", response_model=ApprovalResponse, tags=["Feedback"])
async def approve_output(
    session_id: str,
    request: ApprovalRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Approve agent output and proceed"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    conversation_state = orchestrator.load_session(session_id)
    if not conversation_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Approve output
        success = await orchestrator.approve_output(
            agent_id=request.agent_id,
            phase_id=request.phase_id
        )
        
        # Check if phase can be completed
        if success and request.complete_phase:
            phase_completed = await orchestrator.complete_phase(request.phase_id or conversation_state.current_phase)
            
            if phase_completed:
                # Try to continue to next phase
                await orchestrator.continue_workflow()
        
        # Notify WebSocket clients
        background_tasks.add_task(
            notify_websocket_clients,
            session_id,
            {
                "type": "output_approved",
                "agent_id": request.agent_id,
                "phase_id": request.phase_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return ApprovalResponse(
            agent_id=request.agent_id,
            approved=success,
            next_phase=orchestrator.get_next_phase(),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Approval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve output: {str(e)}")


@app.post("/sessions/{session_id}/phases/{phase_id}/execute", response_model=PhaseExecutionResponse, tags=["Phases"])
async def execute_phase(
    session_id: str,
    phase_id: str,
    request: PhaseExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute entire phase with parallel coordination"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    conversation_state = orchestrator.load_session(session_id)
    if not conversation_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Execute phase with parallel coordination
        agent_tasks = None
        if request.agent_tasks:
            agent_tasks = [task.model_dump() for task in request.agent_tasks]
        
        results = await orchestrator.execute_phase_with_parallel_coordination(
            phase_id=phase_id,
            agent_tasks=agent_tasks
        )
        
        # Notify WebSocket clients
        background_tasks.add_task(
            notify_websocket_clients,
            session_id,
            {
                "type": "phase_completed",
                "phase_id": phase_id,
                "agent_results": {aid: output.to_dict() for aid, output in results.items()},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return PhaseExecutionResponse(
            phase_id=phase_id,
            agent_results={aid: output.to_dict() for aid, output in results.items()},
            execution_time=2.0,  # Placeholder
            success=True,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Phase execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Phase execution failed: {str(e)}")


@app.get("/sessions/{session_id}/context", response_model=ContextResponse, tags=["Context"])
async def get_context(
    session_id: str,
    context_types: Optional[str] = None,
    tags: Optional[str] = None,
    max_results: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve relevant context for session"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    conversation_state = orchestrator.load_session(session_id)
    if not conversation_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Parse context types
        parsed_types = []
        if context_types:
            for ct in context_types.split(','):
                try:
                    parsed_types.append(ContextType(ct.strip()))
                except ValueError:
                    pass
        
        if not parsed_types:
            parsed_types = [ContextType.GLOBAL, ContextType.PHASE, ContextType.AGENT]
        
        # Parse tags
        parsed_tags = set()
        if tags:
            parsed_tags = {tag.strip() for tag in tags.split(',')}
        
        # Retrieve context
        context_data = orchestrator.retrieve_relevant_context(
            context_types=parsed_types,
            tags=parsed_tags if parsed_tags else None,
            max_results=max_results
        )
        
        return ContextResponse(
            session_id=session_id,
            context_data=context_data,
            context_types=[ct.value for ct in parsed_types],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve context: {str(e)}")


@app.get("/statistics", response_model=SystemStatisticsResponse, tags=["System"])
async def get_system_statistics(current_user: dict = Depends(get_current_user)):
    """Get comprehensive system statistics"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    try:
        stats = {
            "parallel_execution": orchestrator.get_parallel_execution_statistics(),
            "decision_engine": orchestrator.get_decision_engine_statistics(),
            "context_manager": orchestrator.get_context_manager_statistics(),
            "error_recovery": orchestrator.get_error_recovery_statistics(),
            "active_sessions": len(orchestrator.active_sessions),
            "websocket_connections": connection_manager.get_connection_count()
        }
        
        return SystemStatisticsResponse(
            statistics=stats,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, session_id)
    
    # Get session logger
    session_logger = get_session_logger(session_id)
    session_logger.log(f"WebSocket connected for session {session_id}")
    
    # Send connection established message
    connection_msg = {
        "type": "connection_established",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "message": "Connected to HAILEI real-time updates"
    }
    await websocket.send_text(json.dumps(connection_msg))
    session_logger.log_websocket_message("SENT", connection_msg)
    
    # Check if session exists and start conversation if needed
    logging.info(f"WebSocket connected for session {session_id}")
    logging.info(f"Orchestrator available: {orchestrator is not None}")
    if orchestrator:
        logging.info(f"Active sessions: {list(orchestrator.active_sessions.keys())}")
        
    if orchestrator and session_id in orchestrator.active_sessions:
        logging.info(f"Session {session_id} found in orchestrator")
        # Get the session state
        session_state = orchestrator.active_sessions[session_id]
        logging.info(f"Session current phase: {session_state.current_phase}")
        
        # If conversation hasn't started yet, start it
        if session_state.current_phase == "course_overview" and len(session_state.conversation_history) == 0:
            try:
                logging.info(f"Starting conversation for session {session_id}")
                
                # Notify that we're starting the conversation
                await websocket.send_text(json.dumps({
                    "type": "conversation_starting",
                    "session_id": session_id,
                    "message": "HAILEI coordinator is preparing your personalized course design experience...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Start the conversation workflow
                result = await orchestrator.process_session(session_id)
                
                # Send the coordinator's greeting message
                await websocket.send_text(json.dumps({
                    "type": "agent_message",
                    "session_id": session_id,
                    "agent_id": "hailei4t_coordinator_agent",
                    "agent_name": "HAILEI Coordinator",
                    "content": result.get("coordinator_response", "Welcome to HAILEI! I'm ready to help you design your course."),
                    "suggestions": [
                        "Tell me more about your course vision",
                        "What's the KDKA framework?",
                        "How does the design process work?",
                        "Let's get started!"
                    ],
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Notify that conversation is now active
                await websocket.send_text(json.dumps({
                    "type": "conversation_started",
                    "session_id": session_id,
                    "current_agent": "hailei4t_coordinator_agent",
                    "current_phase": result.get("current_phase", "course_overview"),
                    "timestamp": datetime.now().isoformat()
                }))
                
            except Exception as e:
                logging.error(f"Failed to start conversation for session {session_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Failed to start conversation: {str(e)}. Please try refreshing the page.",
                    "timestamp": datetime.now().isoformat()
                }))
    else:
        # Session not found, send debug info
        debug_message = f"Session {session_id} not found. Orchestrator: {orchestrator is not None}"
        if orchestrator:
            debug_message += f", Active sessions: {list(orchestrator.active_sessions.keys())}"
        logging.warning(debug_message)
        
        await websocket.send_text(json.dumps({
            "type": "system_message",
            "content": debug_message,
            "timestamp": datetime.now().isoformat()
        }))
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message.get("type") == "subscribe":
                # Subscribe to specific events
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "events": message.get("events", []),
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message.get("type") == "user_message":
                # Handle user messages and forward to orchestrator
                session_logger.log_websocket_message("RECEIVED", message)
                session_logger.log(f"DEBUG: WebSocket received user_message: {message.get('content', '')}")
                logging.info(f"Received user_message: {message}")
                logging.info(f"Orchestrator available: {orchestrator is not None}")
                if orchestrator:
                    logging.info(f"Active sessions: {list(orchestrator.active_sessions.keys())}")
                    logging.info(f"Looking for session: {session_id}")
                
                if orchestrator and session_id in orchestrator.active_sessions:
                    try:
                        # Determine message type from content or explicit type
                        message_type = message.get("message_type", "chat")
                        content = message.get("content", "")
                        
                        # Use LLM for intelligent message type detection
                        session_logger.log(f"DEBUG: About to detect intent for: '{content}'")
                        try:
                            message_type = await detect_message_intent(content)
                            session_logger.log(f"DEBUG: Intent detected as: {message_type}")
                        except Exception as e:
                            session_logger.log(f"ERROR: Intent detection failed: {e}")
                            message_type = "chat"  # fallback
                        
                        # Process user message through orchestrator
                        response = await orchestrator.process_user_message(
                            session_id=session_id,
                            user_message=content,
                            message_type=message_type
                        )
                        
                        # Send response back via WebSocket with enhanced routing
                        response_msg = {
                            "type": "agent_message",
                            "session_id": session_id,
                            "agent_id": response.get("agent", "hailei4t_coordinator_agent"),
                            "agent_name": response.get("agent_name", "HAILEI Coordinator"),
                            "content": response.get("content", ""),
                            "current_phase": response.get("current_phase", "course_overview"),
                            "suggestions": response.get("suggestions", []),
                            "actions_available": [
                                {"type": "approve", "label": "Approve", "enabled": True},
                                {"type": "feedback", "label": "Provide Feedback", "enabled": True},
                                {"type": "question", "label": "Ask Question", "enabled": True}
                            ],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await websocket.send_text(json.dumps(response_msg))
                        session_logger.log_websocket_message("SENT", response_msg)
                        session_logger.log(f"Processed {message_type} message: '{content}'")
                        
                    except Exception as e:
                        logging.error(f"Failed to process user message: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Failed to process message: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Session not found or orchestrator unavailable",
                        "timestamp": datetime.now().isoformat()
                    }))
            
            else:
                # Echo unknown message types
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "original_message": message,
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, session_id)
        logging.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logging.error(f"WebSocket error for session {session_id}: {e}")
        connection_manager.disconnect(websocket, session_id)


# Background tasks
async def setup_session_background(session_id: str, course_request: Dict[str, Any]):
    """Background task for session setup"""
    try:
        # Additional session setup tasks
        logging.info(f"Setting up session {session_id} in background")
        
        # Store initial course context
        if orchestrator:
            orchestrator.store_global_context(
                content={
                    'session_setup': {
                        'session_id': session_id,
                        'course_request': course_request,
                        'setup_timestamp': datetime.now().isoformat()
                    }
                },
                priority=MemoryPriority.MEDIUM,
                tags={'session_setup', session_id}
            )
        
    except Exception as e:
        logging.error(f"Background session setup failed for {session_id}: {e}")


async def notify_websocket_clients(session_id: str, message: Dict[str, Any]):
    """Notify WebSocket clients of events"""
    try:
        await connection_manager.send_to_session(session_id, message)
    except Exception as e:
        logging.error(f"WebSocket notification failed for session {session_id}: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run with uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )