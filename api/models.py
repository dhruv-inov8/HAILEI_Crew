"""
HAILEI API Models

Pydantic models for request/response validation and API documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class UserPreferences(BaseModel):
    """User preferences for conversation and agent behavior"""
    execution_mode_preference: Optional[str] = Field(None, description="Preferred execution mode: parallel, sequential, hybrid")
    preferred_agents: Optional[List[str]] = Field(default_factory=list, description="List of preferred agent IDs")
    communication_style: Optional[str] = Field("professional", description="Communication style preference")
    detail_level: Optional[str] = Field("medium", description="Preferred level of detail: low, medium, high")
    language: Optional[str] = Field("en", description="Preferred language code")


class CourseRequest(BaseModel):
    """Course design request specification"""
    course_title: str = Field(..., description="Title of the course to be designed")
    course_description: Optional[str] = Field(None, description="Detailed course description")
    course_level: str = Field(..., description="Course level: Beginner, Intermediate, Advanced, Graduate")
    course_duration_weeks: int = Field(..., ge=1, le=52, description="Course duration in weeks")
    course_credits: Optional[int] = Field(None, ge=1, le=12, description="Number of academic credits")
    target_audience: str = Field(..., description="Primary target audience for the course")
    prerequisites: Optional[str] = Field(None, description="Required prerequisites for the course")
    learning_outcomes: Optional[List[str]] = Field(default_factory=list, description="Expected learning outcomes")
    special_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Special course requirements")


class CreateSessionRequest(BaseModel):
    """Request to create a new conversation session"""
    course_request: CourseRequest = Field(..., description="Course design requirements")
    session_id: Optional[str] = Field(None, description="Optional custom session ID")
    user_preferences: Optional[UserPreferences] = Field(None, description="User preferences for the session")


class SessionResponse(BaseModel):
    """Response after creating a session"""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session creation status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Session creation timestamp")


class SessionStateResponse(BaseModel):
    """Response containing session state"""
    session_id: str = Field(..., description="Session identifier")
    session_state: Dict[str, Any] = Field(..., description="Complete session state")
    timestamp: datetime = Field(..., description="Response timestamp")


class AgentTask(BaseModel):
    """Task specification for agent execution"""
    agent_id: str = Field(..., description="Agent identifier")
    task_description: str = Field(..., description="Description of the task to perform")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the task")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="List of agent IDs this task depends on")
    priority: Optional[int] = Field(1, ge=1, le=5, description="Task priority (1=low, 5=high)")
    estimated_duration: Optional[float] = Field(60.0, ge=1.0, description="Estimated duration in seconds")


class AgentExecutionRequest(BaseModel):
    """Request to execute a specific agent"""
    task_description: str = Field(..., description="Task for the agent to perform")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for execution")


class AgentExecutionResponse(BaseModel):
    """Response after agent execution"""
    agent_id: str = Field(..., description="Agent identifier")
    output: Dict[str, Any] = Field(..., description="Agent execution output")
    execution_time: float = Field(..., description="Execution time in seconds")
    success: bool = Field(..., description="Execution success status")
    timestamp: datetime = Field(..., description="Execution timestamp")


class FeedbackRequest(BaseModel):
    """Request to submit user feedback"""
    agent_id: str = Field(..., description="Agent ID to provide feedback for")
    feedback: str = Field(..., description="User feedback for refinement")
    phase_id: Optional[str] = Field(None, description="Phase context for feedback")


class FeedbackResponse(BaseModel):
    """Response after processing feedback"""
    feedback_id: str = Field(..., description="Unique feedback identifier")
    status: str = Field(..., description="Processing status")
    refined_output: Optional[Dict[str, Any]] = Field(None, description="Refined agent output if available")
    timestamp: datetime = Field(..., description="Processing timestamp")


class ApprovalRequest(BaseModel):
    """Request to approve agent output"""
    agent_id: str = Field(..., description="Agent ID to approve")
    phase_id: Optional[str] = Field(None, description="Phase context for approval")
    complete_phase: bool = Field(False, description="Whether to complete the phase after approval")


class ApprovalResponse(BaseModel):
    """Response after output approval"""
    agent_id: str = Field(..., description="Agent identifier")
    approved: bool = Field(..., description="Approval status")
    next_phase: Optional[str] = Field(None, description="Next phase in the workflow")
    timestamp: datetime = Field(..., description="Approval timestamp")


class PhaseExecutionRequest(BaseModel):
    """Request to execute an entire phase"""
    agent_tasks: Optional[List[AgentTask]] = Field(None, description="Specific agent tasks for the phase")
    execution_mode: Optional[str] = Field(None, description="Override execution mode: parallel, sequential, hybrid")


class PhaseExecutionResponse(BaseModel):
    """Response after phase execution"""
    phase_id: str = Field(..., description="Phase identifier")
    agent_results: Dict[str, Dict[str, Any]] = Field(..., description="Results from all agents in the phase")
    execution_time: float = Field(..., description="Total execution time in seconds")
    success: bool = Field(..., description="Phase execution success status")
    timestamp: datetime = Field(..., description="Execution timestamp")


class ContextResponse(BaseModel):
    """Response containing relevant context"""
    session_id: str = Field(..., description="Session identifier")
    context_data: List[Dict[str, Any]] = Field(..., description="Retrieved context data")
    context_types: List[str] = Field(..., description="Types of context included")
    timestamp: datetime = Field(..., description="Retrieval timestamp")


class SystemStatisticsResponse(BaseModel):
    """Response containing system statistics"""
    statistics: Dict[str, Any] = Field(..., description="Comprehensive system statistics")
    timestamp: datetime = Field(..., description="Statistics timestamp")


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(..., description="Message type")
    session_id: Optional[str] = Field(None, description="Session identifier")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class ErrorResponse(BaseModel):
    """Error response structure"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    session_id: Optional[str] = Field(None, description="Associated session ID")


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    WORKING = "working"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    ERROR = "error"


class PhaseStatus(str, Enum):
    """Phase status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    ERROR = "error"


class ExecutionMode(str, Enum):
    """Execution mode enumeration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class ContextType(str, Enum):
    """Context type enumeration"""
    GLOBAL = "global"
    PHASE = "phase"
    AGENT = "agent"
    USER = "user"
    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"


# Health check models
class ComponentHealth(BaseModel):
    """Health status of individual component"""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    last_check: datetime = Field(..., description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class HealthResponse(BaseModel):
    """System health check response"""
    status: str = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Individual component health statuses")
    statistics: Optional[Dict[str, Any]] = Field(None, description="System statistics")


# Authentication models
class TokenRequest(BaseModel):
    """Authentication token request"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class UserInfo(BaseModel):
    """User information"""
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    role: str = Field("user", description="User role")
    permissions: List[str] = Field(default_factory=list, description="User permissions")


# Batch operation models
class BatchAgentExecutionRequest(BaseModel):
    """Request to execute multiple agents"""
    agent_tasks: List[AgentTask] = Field(..., description="List of agent tasks to execute")
    execution_mode: Optional[str] = Field("parallel", description="Execution mode for the batch")
    max_parallel: Optional[int] = Field(3, ge=1, le=10, description="Maximum parallel executions")


class BatchAgentExecutionResponse(BaseModel):
    """Response after batch agent execution"""
    batch_id: str = Field(..., description="Unique batch execution identifier")
    results: Dict[str, Dict[str, Any]] = Field(..., description="Results from all agents")
    summary: Dict[str, Any] = Field(..., description="Execution summary statistics")
    timestamp: datetime = Field(..., description="Execution timestamp")


# Export/Import models
class ExportRequest(BaseModel):
    """Request to export session data"""
    session_id: str = Field(..., description="Session to export")
    include_context: bool = Field(True, description="Include context data in export")
    include_history: bool = Field(True, description="Include conversation history")
    format: str = Field("json", description="Export format: json, yaml")


class ExportResponse(BaseModel):
    """Response containing exported data"""
    export_id: str = Field(..., description="Unique export identifier")
    download_url: str = Field(..., description="URL to download exported data")
    expires_at: datetime = Field(..., description="Download URL expiration time")
    file_size: int = Field(..., description="Exported file size in bytes")


class ImportRequest(BaseModel):
    """Request to import session data"""
    data: Dict[str, Any] = Field(..., description="Data to import")
    merge_strategy: str = Field("replace", description="Import merge strategy: replace, merge, skip")
    target_session_id: Optional[str] = Field(None, description="Target session ID for import")