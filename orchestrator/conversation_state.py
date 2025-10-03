"""
HAILEI Conversational State Management System

Manages conversation state, phase tracking, and agent outputs for frontend deployment.
Designed for Flask/FastAPI integration with WebSocket support.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid


class PhaseStatus(Enum):
    """Phase execution status for frontend progress tracking"""
    PENDING = "pending"
    ACTIVE = "active" 
    REFINEMENT = "refinement"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class AgentStatus(Enum):
    """Agent execution status for real-time updates"""
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentOutput:
    """Structured agent output for frontend display and iteration"""
    agent_id: str
    agent_name: str
    phase: str
    content: str
    timestamp: datetime
    version: int = 1
    refinement_notes: List[str] = field(default_factory=list)
    user_feedback: List[str] = field(default_factory=list)
    is_approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'phase': self.phase,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'refinement_notes': self.refinement_notes,
            'user_feedback': self.user_feedback,
            'is_approved': self.is_approved,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentOutput':
        """Create from dictionary for API deserialization"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PhaseState:
    """Phase execution state for workflow management"""
    phase_id: str
    phase_name: str
    status: PhaseStatus
    assigned_agents: List[str]
    active_agents: List[str] = field(default_factory=list)
    completed_agents: List[str] = field(default_factory=list)
    outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_approvals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'phase_id': self.phase_id,
            'phase_name': self.phase_name,
            'status': self.status.value,
            'assigned_agents': self.assigned_agents,
            'active_agents': self.active_agents,
            'completed_agents': self.completed_agents,
            'outputs': {k: v.to_dict() for k, v in self.outputs.items()},
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'user_approvals': self.user_approvals,
            'dependencies': self.dependencies
        }


class ConversationState:
    """
    Central conversation state management for HAILEI orchestrator.
    
    Features:
    - Thread-safe state management for concurrent access
    - Frontend progress tracking with real-time updates
    - Conversation history with full context preservation
    - Agent output versioning for iterative refinement
    - WebSocket-ready state serialization
    - Session persistence for long-running conversations
    """
    
    def __init__(self, session_id: Optional[str] = None, course_request: Optional[Dict] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Course context
        self.course_request = course_request or {}
        self.frameworks = {
            'kdka': {},
            'prrr': {}
        }
        
        # Conversation tracking
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_phase: Optional[str] = None
        self.active_agent: Optional[str] = None
        
        # Phase management
        self.phases: Dict[str, PhaseState] = {}
        self.phase_order: List[str] = [
            'course_overview',
            'foundation_design', 
            'content_creation',
            'technical_design',
            'quality_review',
            'ethical_audit',
            'final_integration'
        ]
        
        # Agent state tracking
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_outputs: Dict[str, List[AgentOutput]] = {}
        
        # User interaction state
        self.user_approvals: Dict[str, bool] = {}
        self.pending_user_input: Optional[Dict[str, Any]] = None
        self.refinement_cycles: Dict[str, int] = {}
        
        # Context memory for agents
        self.context_memory: Dict[str, Any] = {
            'global_context': {},
            'agent_contexts': {},
            'phase_contexts': {}
        }
        
        # Initialize phases
        self._initialize_phases()
        
        # Set initial phase to course_overview
        self.current_phase = 'course_overview'
        if 'course_overview' in self.phases:
            self.phases['course_overview'].status = PhaseStatus.ACTIVE
    
    def _initialize_phases(self):
        """Initialize phase structure for workflow management"""
        phase_definitions = {
            'course_overview': {
                'name': 'Course Overview & Approval',
                'agents': ['coordinator'],
                'dependencies': []
            },
            'foundation_design': {
                'name': 'Course Foundation Design',
                'agents': ['ipdai'],
                'dependencies': ['course_overview']
            },
            'content_creation': {
                'name': 'Instructional Content Creation', 
                'agents': ['cauthai'],
                'dependencies': ['foundation_design']
            },
            'technical_design': {
                'name': 'LMS Technical Design',
                'agents': ['tfdai'],
                'dependencies': ['content_creation']
            },
            'quality_review': {
                'name': 'Quality Review & Enhancement',
                'agents': ['editorai'],
                'dependencies': ['technical_design']
            },
            'ethical_audit': {
                'name': 'Ethical Compliance Audit',
                'agents': ['ethosai'],
                'dependencies': ['quality_review']
            },
            'final_integration': {
                'name': 'Final Integration & Delivery',
                'agents': ['coordinator'],
                'dependencies': ['ethical_audit']
            }
        }
        
        for phase_id, definition in phase_definitions.items():
            self.phases[phase_id] = PhaseState(
                phase_id=phase_id,
                phase_name=definition['name'],
                status=PhaseStatus.PENDING,
                assigned_agents=definition['agents'],
                dependencies=definition['dependencies']
            )
    
    def add_conversation_turn(self, speaker: str, message: str, metadata: Optional[Dict] = None):
        """Add conversation turn with timestamp and metadata"""
        turn = {
            'speaker': speaker,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(turn)
        self.last_activity = datetime.now()
    
    def set_current_phase(self, phase_id: str):
        """Set current active phase with state updates"""
        if phase_id in self.phases:
            # Complete previous phase if exists
            if self.current_phase and self.current_phase != phase_id:
                prev_phase = self.phases[self.current_phase]
                if prev_phase.status == PhaseStatus.ACTIVE:
                    prev_phase.status = PhaseStatus.COMPLETED
                    prev_phase.completed_at = datetime.now()
            
            # Activate new phase
            self.current_phase = phase_id
            current_phase = self.phases[phase_id]
            current_phase.status = PhaseStatus.ACTIVE
            current_phase.started_at = datetime.now()
            
            self.last_activity = datetime.now()
    
    def set_active_agent(self, agent_id: str, status: AgentStatus = AgentStatus.WORKING):
        """Set currently active agent with status tracking"""
        self.active_agent = agent_id
        
        if agent_id not in self.agents:
            self.agents[agent_id] = {}
        
        self.agents[agent_id]['status'] = status.value
        self.agents[agent_id]['last_activity'] = datetime.now().isoformat()
        self.last_activity = datetime.now()
    
    def add_agent_output(self, output: AgentOutput):
        """Add agent output with versioning and phase tracking"""
        agent_id = output.agent_id
        
        # Initialize agent output history
        if agent_id not in self.agent_outputs:
            self.agent_outputs[agent_id] = []
        
        # Set version number
        existing_outputs = [o for o in self.agent_outputs[agent_id] if o.phase == output.phase]
        output.version = len(existing_outputs) + 1
        
        # Add to agent outputs
        self.agent_outputs[agent_id].append(output)
        
        # Add to current phase outputs
        if self.current_phase and self.current_phase in self.phases:
            self.phases[self.current_phase].outputs[agent_id] = output
        
        self.last_activity = datetime.now()
    
    def get_latest_output(self, agent_id: str, phase: Optional[str] = None) -> Optional[AgentOutput]:
        """Get latest output from agent, optionally filtered by phase"""
        if agent_id not in self.agent_outputs:
            return None
        
        outputs = self.agent_outputs[agent_id]
        if phase:
            outputs = [o for o in outputs if o.phase == phase]
        
        return outputs[-1] if outputs else None
    
    def approve_output(self, agent_id: str, phase: Optional[str] = None):
        """Mark agent output as approved by user"""
        output = self.get_latest_output(agent_id, phase)
        if output:
            output.is_approved = True
            self.user_approvals[f"{agent_id}_{phase or self.current_phase}"] = True
            self.last_activity = datetime.now()
    
    def add_user_feedback(self, agent_id: str, feedback: str, phase: Optional[str] = None):
        """Add user feedback to agent output for refinement"""
        output = self.get_latest_output(agent_id, phase)
        if output:
            output.user_feedback.append(feedback)
            
            # Increment refinement cycle
            key = f"{agent_id}_{phase or self.current_phase}"
            self.refinement_cycles[key] = self.refinement_cycles.get(key, 0) + 1
            
            self.last_activity = datetime.now()
    
    def set_pending_user_input(self, input_data: Dict[str, Any]):
        """Set pending user input for frontend handling"""
        self.pending_user_input = input_data
        self.last_activity = datetime.now()
    
    def clear_pending_user_input(self):
        """Clear pending user input after handling"""
        self.pending_user_input = None
        self.last_activity = datetime.now()
    
    def update_context_memory(self, key: str, value: Any, scope: str = 'global'):
        """Update context memory for agent access"""
        if scope == 'global':
            self.context_memory['global_context'][key] = value
        elif scope == 'agent' and self.active_agent:
            if self.active_agent not in self.context_memory['agent_contexts']:
                self.context_memory['agent_contexts'][self.active_agent] = {}
            self.context_memory['agent_contexts'][self.active_agent][key] = value
        elif scope == 'phase' and self.current_phase:
            if self.current_phase not in self.context_memory['phase_contexts']:
                self.context_memory['phase_contexts'][self.current_phase] = {}
            self.context_memory['phase_contexts'][self.current_phase][key] = value
        
        self.last_activity = datetime.now()
    
    def get_context_memory(self, key: str, scope: str = 'global') -> Any:
        """Retrieve context memory for agent access"""
        if scope == 'global':
            return self.context_memory['global_context'].get(key)
        elif scope == 'agent' and self.active_agent:
            agent_context = self.context_memory['agent_contexts'].get(self.active_agent, {})
            return agent_context.get(key)
        elif scope == 'phase' and self.current_phase:
            phase_context = self.context_memory['phase_contexts'].get(self.current_phase, {})
            return phase_context.get(key)
        return None
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get phase progress for frontend display"""
        completed = len([p for p in self.phases.values() if p.status == PhaseStatus.COMPLETED])
        total = len(self.phases)
        
        return {
            'current_phase': self.current_phase,
            'completed_phases': completed,
            'total_phases': total,
            'progress_percentage': int((completed / total) * 100),
            'phases': {pid: phase.to_dict() for pid, phase in self.phases.items()}
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation state for API/WebSocket transmission"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'course_request': self.course_request,
            'frameworks': self.frameworks,
            'conversation_history': self.conversation_history,
            'current_phase': self.current_phase,
            'active_agent': self.active_agent,
            'phases': {pid: phase.to_dict() for pid, phase in self.phases.items()},
            'agents': self.agents,
            'agent_outputs': {
                agent_id: [output.to_dict() for output in outputs]
                for agent_id, outputs in self.agent_outputs.items()
            },
            'user_approvals': self.user_approvals,
            'pending_user_input': self.pending_user_input,
            'refinement_cycles': self.refinement_cycles,
            'context_memory': self.context_memory,
            'progress': self.get_phase_progress()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Deserialize conversation state from API/storage"""
        state = cls(session_id=data['session_id'], course_request=data['course_request'])
        
        state.created_at = datetime.fromisoformat(data['created_at'])
        state.last_activity = datetime.fromisoformat(data['last_activity'])
        state.frameworks = data['frameworks']
        state.conversation_history = data['conversation_history']
        state.current_phase = data['current_phase']
        state.active_agent = data['active_agent']
        state.agents = data['agents']
        state.user_approvals = data['user_approvals']
        state.pending_user_input = data['pending_user_input']
        state.refinement_cycles = data['refinement_cycles']
        state.context_memory = data['context_memory']
        
        # Restore agent outputs
        for agent_id, outputs_data in data['agent_outputs'].items():
            state.agent_outputs[agent_id] = [
                AgentOutput.from_dict(output_data) for output_data in outputs_data
            ]
        
        return state
    
    def __repr__(self) -> str:
        return f"ConversationState(session_id='{self.session_id}', phase='{self.current_phase}', agent='{self.active_agent}')"