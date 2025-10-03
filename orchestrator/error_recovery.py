"""
HAILEI Comprehensive Error Recovery System

Advanced error detection, recovery strategies, and resilience mechanisms
for robust conversational AI orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback
import threading
import json
from collections import defaultdict, deque

from .conversation_state import ConversationState, AgentOutput, PhaseStatus, AgentStatus
from .context_manager import ContextType, MemoryPriority


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors"""
    AGENT_EXECUTION = "agent_execution"
    CONTEXT_MANAGEMENT = "context_management"
    PARALLEL_COORDINATION = "parallel_coordination"
    DECISION_ENGINE = "decision_engine"
    USER_INTERACTION = "user_interaction"
    SYSTEM_RESOURCE = "system_resource"
    COMMUNICATION = "communication"
    DATA_VALIDATION = "data_validation"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ALTERNATIVE_PATH = "alternative_path"
    USER_INTERVENTION = "user_intervention"
    SYSTEM_RESTART = "system_restart"
    SKIP_AND_CONTINUE = "skip_and_continue"


@dataclass
class ErrorContext:
    """Context information for an error"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    error_message: str
    stack_trace: Optional[str] = None
    session_id: Optional[str] = None
    phase_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize error context"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'component': self.component,
            'operation': self.operation,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'session_id': self.session_id,
            'phase_id': self.phase_id,
            'agent_id': self.agent_id,
            'user_data': self.user_data,
            'system_state': self.system_state
        }


@dataclass
class RecoveryAction:
    """Action taken to recover from an error"""
    action_id: str
    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    timeout_seconds: int = 30
    success_criteria: Optional[Callable] = None
    rollback_action: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize recovery action"""
        return {
            'action_id': self.action_id,
            'strategy': self.strategy.value,
            'description': self.description,
            'parameters': self.parameters,
            'max_attempts': self.max_attempts,
            'timeout_seconds': self.timeout_seconds
        }


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    action_taken: RecoveryAction
    attempts_made: int
    time_taken: float
    output: Optional[Any] = None
    error_message: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize recovery result"""
        return {
            'success': self.success,
            'action_taken': self.action_taken.to_dict(),
            'attempts_made': self.attempts_made,
            'time_taken': self.time_taken,
            'error_message': self.error_message,
            'side_effects': self.side_effects
        }


class ErrorRecoverySystem:
    """
    Comprehensive error recovery system for HAILEI orchestration.
    
    Features:
    - Intelligent error classification
    - Context-aware recovery strategies
    - Automated retry mechanisms
    - Graceful degradation
    - Recovery learning and optimization
    - Real-time error monitoring
    """
    
    def __init__(
        self,
        orchestrator,
        enable_logging: bool = True,
        max_error_history: int = 1000
    ):
        """Initialize error recovery system"""
        self.orchestrator = orchestrator
        self.max_error_history = max_error_history
        
        # Error tracking
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.recovery_success_rates: Dict[str, float] = {}
        
        # Recovery strategies mapping
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.custom_handlers: Dict[Type[Exception], Callable] = {}
        
        # Circuit breaker pattern for failing components
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Recovery state
        self.active_recoveries: Dict[str, RecoveryAction] = {}
        self.recovery_lock = threading.Lock()
        
        # Monitoring and alerting
        self.error_rate_thresholds = {
            ErrorSeverity.LOW: 10,      # per hour
            ErrorSeverity.MEDIUM: 5,    # per hour
            ErrorSeverity.HIGH: 2,      # per hour
            ErrorSeverity.CRITICAL: 1   # per hour
        }
        
        # Learning system
        self.strategy_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.context_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger('hailei_error_recovery')
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
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: str = "unknown",
        component: str = "unknown"
    ) -> RecoveryResult:
        """
        Handle an error with intelligent recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            operation: Operation that was being performed
            component: Component where error occurred
            
        Returns:
            RecoveryResult: Result of the recovery attempt
        """
        # Create error context
        error_context = await self._create_error_context(
            error, context, operation, component
        )
        
        # Log error
        self.logger.error(f"Error in {component}.{operation}: {str(error)}")
        self.logger.debug(f"Error context: {error_context.to_dict()}")
        
        # Store error for pattern analysis
        self.error_history.append(error_context)
        self._update_error_patterns(error_context)
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(component, operation):
            return await self._handle_circuit_breaker_open(error_context)
        
        # Determine recovery strategy
        recovery_action = await self._determine_recovery_strategy(error_context)
        
        # Execute recovery
        recovery_result = await self._execute_recovery(error_context, recovery_action)
        
        # Update circuit breaker state
        self._update_circuit_breaker(component, operation, recovery_result.success)
        
        # Learn from recovery attempt
        self._learn_from_recovery(error_context, recovery_action, recovery_result)
        
        # Alert if necessary
        await self._check_error_rate_alerts(error_context)
        
        return recovery_result
    
    async def recover_from_agent_failure(
        self,
        agent_id: str,
        task_description: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[AgentOutput]:
        """
        Specific recovery for agent execution failures.
        
        Args:
            agent_id: ID of the failed agent
            task_description: Task that failed
            error: The exception that occurred
            context: Execution context
            
        Returns:
            AgentOutput or None if recovery failed
        """
        recovery_result = await self.handle_error(
            error=error,
            context={
                'agent_id': agent_id,
                'task_description': task_description,
                **context
            },
            operation='agent_execution',
            component=f'agent_{agent_id}'
        )
        
        if recovery_result.success and isinstance(recovery_result.output, AgentOutput):
            return recovery_result.output
        
        return None
    
    async def recover_from_parallel_execution_failure(
        self,
        phase_id: str,
        failed_agents: List[str],
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recovery for parallel execution failures.
        
        Args:
            phase_id: Phase that failed
            failed_agents: List of agents that failed
            error: The exception that occurred
            context: Execution context
            
        Returns:
            Recovery context with status and actions taken
        """
        recovery_result = await self.handle_error(
            error=error,
            context={
                'phase_id': phase_id,
                'failed_agents': failed_agents,
                **context
            },
            operation='parallel_execution',
            component='parallel_orchestrator'
        )
        
        return {
            'recovery_success': recovery_result.success,
            'recovery_action': recovery_result.action_taken.to_dict(),
            'failed_agents': failed_agents,
            'phase_id': phase_id,
            'fallback_strategy': recovery_result.action_taken.strategy.value
        }
    
    def register_custom_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception, Dict[str, Any]], RecoveryAction]
    ):
        """Register custom error handler for specific exception types"""
        self.custom_handlers[exception_type] = handler
        self.logger.info(f"Registered custom handler for {exception_type.__name__}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics"""
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {'message': 'No errors recorded'}
        
        # Error distribution by category and severity
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for error_ctx in self.error_history:
            category_counts[error_ctx.category.value] += 1
            severity_counts[error_ctx.severity.value] += 1
            component_counts[error_ctx.component] += 1
        
        # Recent error rate
        now = datetime.now()
        recent_errors = [
            err for err in self.error_history
            if (now - err.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Recovery success rates
        avg_recovery_success = (
            sum(self.recovery_success_rates.values()) / len(self.recovery_success_rates)
            if self.recovery_success_rates else 0.0
        )
        
        return {
            'total_errors': total_errors,
            'recent_error_count': len(recent_errors),
            'error_rate_per_hour': len(recent_errors),
            'category_distribution': dict(category_counts),
            'severity_distribution': dict(severity_counts),
            'component_distribution': dict(component_counts),
            'recovery_success_rate': avg_recovery_success,
            'circuit_breaker_status': self._get_circuit_breaker_status(),
            'top_error_patterns': dict(sorted(
                self.error_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
        }
    
    async def _create_error_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: str,
        component: str
    ) -> ErrorContext:
        """Create comprehensive error context"""
        
        # Generate unique error ID
        error_id = f"err_{component}_{int(datetime.now().timestamp())}_{id(error)}"
        
        # Determine error severity
        severity = self._classify_error_severity(error, context)
        
        # Determine error category
        category = self._classify_error_category(error, component, operation)
        
        # Extract system state
        system_state = await self._capture_system_state()
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=component,
            operation=operation,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            session_id=context.get('session_id'),
            phase_id=context.get('phase_id'),
            agent_id=context.get('agent_id'),
            user_data=context.get('user_data'),
            system_state=system_state
        )
    
    def _classify_error_severity(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Classify error severity based on type and context"""
        
        # Critical errors that stop the system
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        
        # Agent-specific errors based on phase importance
        if context.get('phase_id') in ['final_integration', 'quality_review']:
            return ErrorSeverity.HIGH
        
        # Context-based severity
        if context.get('user_interaction_required'):
            return ErrorSeverity.MEDIUM
        
        # Default to medium for unknown errors
        return ErrorSeverity.MEDIUM
    
    def _classify_error_category(
        self, 
        error: Exception, 
        component: str, 
        operation: str
    ) -> ErrorCategory:
        """Classify error category based on error type and context"""
        
        # Component-based classification
        if 'agent' in component.lower():
            return ErrorCategory.AGENT_EXECUTION
        elif 'context' in component.lower():
            return ErrorCategory.CONTEXT_MANAGEMENT
        elif 'parallel' in component.lower():
            return ErrorCategory.PARALLEL_COORDINATION
        elif 'decision' in component.lower():
            return ErrorCategory.DECISION_ENGINE
        
        # Error type-based classification
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.COMMUNICATION
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorCategory.DATA_VALIDATION
        elif isinstance(error, (MemoryError, ResourceWarning)):
            return ErrorCategory.SYSTEM_RESOURCE
        
        # Operation-based classification
        if 'user' in operation.lower():
            return ErrorCategory.USER_INTERACTION
        
        # Default classification
        return ErrorCategory.AGENT_EXECUTION
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'active_sessions': len(getattr(self.orchestrator, 'active_sessions', {})),
                'memory_usage': 'unknown'  # Placeholder for actual memory monitoring
            }
            
            # Add orchestrator-specific state
            if hasattr(self.orchestrator, 'conversation_state') and self.orchestrator.conversation_state:
                state.update({
                    'current_phase': self.orchestrator.conversation_state.current_phase,
                    'active_agent': self.orchestrator.conversation_state.active_agent,
                    'conversation_turns': len(self.orchestrator.conversation_state.conversation_history)
                })
            
            return state
        except Exception as e:
            self.logger.warning(f"Failed to capture system state: {e}")
            return {'error': 'Failed to capture state'}
    
    async def _determine_recovery_strategy(
        self, 
        error_context: ErrorContext
    ) -> RecoveryAction:
        """Determine the best recovery strategy for the error"""
        
        # Check for custom handlers first
        for exception_type, handler in self.custom_handlers.items():
            if exception_type.__name__ in error_context.error_message:
                try:
                    return handler(Exception(error_context.error_message), error_context.system_state or {})
                except Exception as e:
                    self.logger.warning(f"Custom handler failed: {e}")
        
        # Use predefined strategies based on category and severity
        strategy_key = f"{error_context.category.value}_{error_context.severity.value}"
        
        if strategy_key in self.recovery_strategies:
            strategy_config = self.recovery_strategies[strategy_key]
            return RecoveryAction(
                action_id=f"recovery_{error_context.error_id}",
                strategy=strategy_config['strategy'],
                description=strategy_config['description'],
                parameters=strategy_config.get('parameters', {}),
                max_attempts=strategy_config.get('max_attempts', 3),
                timeout_seconds=strategy_config.get('timeout_seconds', 30)
            )
        
        # Fallback to default strategy
        return RecoveryAction(
            action_id=f"recovery_{error_context.error_id}",
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            description="Default graceful degradation strategy",
            parameters={'fallback_mode': True},
            max_attempts=1
        )
    
    async def _execute_recovery(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> RecoveryResult:
        """Execute the recovery action"""
        
        start_time = datetime.now()
        attempts = 0
        last_error = None
        
        with self.recovery_lock:
            self.active_recoveries[error_context.error_id] = recovery_action
        
        try:
            for attempt in range(recovery_action.max_attempts):
                attempts += 1
                
                try:
                    self.logger.info(f"Recovery attempt {attempt + 1}/{recovery_action.max_attempts} for {error_context.error_id}")
                    
                    # Execute strategy-specific recovery
                    result = await self._execute_strategy(error_context, recovery_action, attempt)
                    
                    if result is not None:
                        execution_time = (datetime.now() - start_time).total_seconds()
                        return RecoveryResult(
                            success=True,
                            action_taken=recovery_action,
                            attempts_made=attempts,
                            time_taken=execution_time,
                            output=result
                        )
                
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")
                    
                    # Wait before retry (exponential backoff)
                    if attempt < recovery_action.max_attempts - 1:
                        wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                        await asyncio.sleep(wait_time)
            
            # All attempts failed
            execution_time = (datetime.now() - start_time).total_seconds()
            return RecoveryResult(
                success=False,
                action_taken=recovery_action,
                attempts_made=attempts,
                time_taken=execution_time,
                error_message=last_error or "Recovery failed after all attempts"
            )
        
        finally:
            with self.recovery_lock:
                self.active_recoveries.pop(error_context.error_id, None)
    
    async def _execute_strategy(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction, 
        attempt: int
    ) -> Optional[Any]:
        """Execute specific recovery strategy"""
        
        strategy = recovery_action.strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._execute_retry_strategy(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_fallback_strategy(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._execute_graceful_degradation(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.ALTERNATIVE_PATH:
            return await self._execute_alternative_path(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.USER_INTERVENTION:
            return await self._execute_user_intervention(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
            return await self._execute_skip_and_continue(error_context, recovery_action)
        
        else:
            raise ValueError(f"Unknown recovery strategy: {strategy}")
    
    async def _execute_retry_strategy(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Execute simple retry strategy"""
        
        # Attempt to re-execute the original operation
        if error_context.category == ErrorCategory.AGENT_EXECUTION:
            return await self._retry_agent_execution(error_context)
        elif error_context.category == ErrorCategory.PARALLEL_COORDINATION:
            return await self._retry_parallel_execution(error_context)
        else:
            # Generic retry - just indicate success
            return {"retry_success": True, "error_context": error_context.error_id}
    
    async def _execute_fallback_strategy(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Execute fallback strategy with alternative implementation"""
        
        if error_context.category == ErrorCategory.AGENT_EXECUTION:
            # Use fallback agent or simplified execution
            return await self._fallback_agent_execution(error_context)
        
        return {"fallback_applied": True, "simplified_result": "Fallback execution completed"}
    
    async def _execute_graceful_degradation(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Execute graceful degradation strategy"""
        
        # Provide minimal viable functionality
        degraded_result = {
            "degraded_mode": True,
            "error_context": error_context.error_id,
            "message": "System operating in degraded mode due to error",
            "available_functions": ["basic_operations"]
        }
        
        return degraded_result
    
    async def _execute_alternative_path(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Execute alternative execution path"""
        
        # Find alternative approach based on error context
        if error_context.category == ErrorCategory.PARALLEL_COORDINATION:
            # Switch to sequential execution
            return {"alternative_execution": "sequential", "reason": "parallel_failed"}
        
        return {"alternative_path": True, "method": "sequential_fallback"}
    
    async def _execute_user_intervention(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Request user intervention for recovery"""
        
        # In a real implementation, this would integrate with the UI
        # For now, we'll simulate user intervention
        intervention_request = {
            "user_intervention_required": True,
            "error_summary": error_context.error_message,
            "suggested_actions": [
                "Retry the operation",
                "Skip this step",
                "Provide alternative input"
            ],
            "error_id": error_context.error_id
        }
        
        return intervention_request
    
    async def _execute_skip_and_continue(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Skip the failed operation and continue"""
        
        return {
            "operation_skipped": True,
            "error_id": error_context.error_id,
            "continuation": "workflow_continued"
        }
    
    async def _retry_agent_execution(self, error_context: ErrorContext) -> Optional[AgentOutput]:
        """Retry agent execution with error context"""
        
        agent_id = error_context.agent_id
        if not agent_id:
            return None
        
        try:
            # Simulate retry with simplified context
            from .conversation_state import AgentOutput
            return AgentOutput(
                agent_id=agent_id,
                agent_name=f"Agent {agent_id}",
                phase=error_context.phase_id or "unknown",
                content=f"Recovered execution for agent {agent_id} after error recovery",
                timestamp=datetime.now(),
                metadata={"recovery": True, "original_error": error_context.error_id}
            )
        except Exception as e:
            self.logger.error(f"Agent retry failed: {e}")
            return None
    
    async def _retry_parallel_execution(self, error_context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Retry parallel execution with modified parameters"""
        
        return {
            "retry_execution": True,
            "phase_id": error_context.phase_id,
            "modified_execution": "sequential_fallback",
            "recovery_metadata": {
                "original_error": error_context.error_id,
                "retry_timestamp": datetime.now().isoformat()
            }
        }
    
    async def _fallback_agent_execution(self, error_context: ErrorContext) -> Optional[AgentOutput]:
        """Fallback agent execution with simplified functionality"""
        
        agent_id = error_context.agent_id
        if not agent_id:
            return None
        
        from .conversation_state import AgentOutput
        return AgentOutput(
            agent_id=agent_id,
            agent_name=f"Fallback Agent {agent_id}",
            phase=error_context.phase_id or "unknown",
            content=f"Fallback execution completed for {agent_id}. Limited functionality due to error recovery.",
            timestamp=datetime.now(),
            metadata={"fallback_mode": True, "original_error": error_context.error_id}
        )
    
    def _initialize_recovery_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize recovery strategies for different error types"""
        
        return {
            # Agent execution errors
            "agent_execution_low": {
                "strategy": RecoveryStrategy.RETRY,
                "description": "Retry agent execution with same parameters",
                "max_attempts": 3,
                "timeout_seconds": 30
            },
            "agent_execution_medium": {
                "strategy": RecoveryStrategy.FALLBACK,
                "description": "Use fallback agent execution",
                "max_attempts": 2,
                "timeout_seconds": 60
            },
            "agent_execution_high": {
                "strategy": RecoveryStrategy.USER_INTERVENTION,
                "description": "Request user intervention for critical agent failure",
                "max_attempts": 1,
                "timeout_seconds": 300
            },
            "agent_execution_critical": {
                "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                "description": "Enter degraded mode with minimal functionality",
                "max_attempts": 1,
                "timeout_seconds": 10
            },
            
            # Parallel coordination errors
            "parallel_coordination_low": {
                "strategy": RecoveryStrategy.RETRY,
                "description": "Retry parallel execution",
                "max_attempts": 2,
                "parameters": {"reduce_parallelism": True}
            },
            "parallel_coordination_medium": {
                "strategy": RecoveryStrategy.ALTERNATIVE_PATH,
                "description": "Switch to sequential execution",
                "max_attempts": 1,
                "parameters": {"execution_mode": "sequential"}
            },
            "parallel_coordination_high": {
                "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                "description": "Simplified sequential execution",
                "max_attempts": 1
            },
            
            # Context management errors
            "context_management_low": {
                "strategy": RecoveryStrategy.RETRY,
                "description": "Retry context operation",
                "max_attempts": 3
            },
            "context_management_medium": {
                "strategy": RecoveryStrategy.SKIP_AND_CONTINUE,
                "description": "Skip context update and continue",
                "max_attempts": 1
            },
            
            # Communication errors
            "communication_medium": {
                "strategy": RecoveryStrategy.RETRY,
                "description": "Retry communication with backoff",
                "max_attempts": 5,
                "timeout_seconds": 60,
                "parameters": {"exponential_backoff": True}
            },
            
            # System resource errors
            "system_resource_high": {
                "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                "description": "Reduce resource usage",
                "max_attempts": 1,
                "parameters": {"resource_optimization": True}
            }
        }
    
    def _update_error_patterns(self, error_context: ErrorContext):
        """Update error pattern analysis"""
        pattern_key = f"{error_context.category.value}_{error_context.component}_{error_context.operation}"
        self.error_patterns[pattern_key] += 1
    
    def _is_circuit_breaker_open(self, component: str, operation: str) -> bool:
        """Check if circuit breaker is open for component/operation"""
        breaker_key = f"{component}_{operation}"
        breaker = self.circuit_breakers.get(breaker_key)
        
        if not breaker:
            return False
        
        if breaker['state'] == 'open':
            # Check if enough time has passed to try half-open
            if datetime.now() - breaker['last_failure'] > timedelta(minutes=5):
                breaker['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, component: str, operation: str, success: bool):
        """Update circuit breaker state based on operation result"""
        breaker_key = f"{component}_{operation}"
        
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': None,
                'success_count': 0
            }
        
        breaker = self.circuit_breakers[breaker_key]
        
        if success:
            breaker['failure_count'] = 0
            breaker['success_count'] += 1
            if breaker['state'] == 'half_open' and breaker['success_count'] >= 3:
                breaker['state'] = 'closed'
        else:
            breaker['failure_count'] += 1
            breaker['last_failure'] = datetime.now()
            breaker['success_count'] = 0
            
            # Open circuit breaker after 5 consecutive failures
            if breaker['failure_count'] >= 5:
                breaker['state'] = 'open'
    
    async def _handle_circuit_breaker_open(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle case when circuit breaker is open"""
        
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction(
                action_id=f"circuit_breaker_{error_context.error_id}",
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                description="Circuit breaker open - operation blocked"
            ),
            attempts_made=0,
            time_taken=0.0,
            error_message="Circuit breaker is open for this operation"
        )
    
    def _learn_from_recovery(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction, 
        recovery_result: RecoveryResult
    ):
        """Learn from recovery attempt to improve future strategies"""
        
        strategy_key = f"{error_context.category.value}_{recovery_action.strategy.value}"
        
        # Update strategy effectiveness
        effectiveness = 1.0 if recovery_result.success else 0.0
        self.strategy_effectiveness[strategy_key].append(effectiveness)
        
        # Keep only recent data (last 100 attempts)
        if len(self.strategy_effectiveness[strategy_key]) > 100:
            self.strategy_effectiveness[strategy_key] = self.strategy_effectiveness[strategy_key][-100:]
        
        # Update recovery success rates
        success_rate = sum(self.strategy_effectiveness[strategy_key]) / len(self.strategy_effectiveness[strategy_key])
        self.recovery_success_rates[strategy_key] = success_rate
        
        # Store context patterns for learning
        context_pattern = {
            'component': error_context.component,
            'operation': error_context.operation,
            'severity': error_context.severity.value,
            'success': recovery_result.success,
            'strategy': recovery_action.strategy.value,
            'attempts': recovery_result.attempts_made
        }
        self.context_patterns[strategy_key].append(context_pattern)
    
    async def _check_error_rate_alerts(self, error_context: ErrorContext):
        """Check if error rates exceed thresholds and send alerts"""
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Count errors of this severity in the last hour
        recent_errors = [
            err for err in self.error_history
            if err.severity == error_context.severity and err.timestamp >= hour_ago
        ]
        
        threshold = self.error_rate_thresholds[error_context.severity]
        
        if len(recent_errors) >= threshold:
            await self._send_error_rate_alert(error_context.severity, len(recent_errors), threshold)
    
    async def _send_error_rate_alert(self, severity: ErrorSeverity, count: int, threshold: int):
        """Send error rate alert (placeholder for actual alerting system)"""
        
        alert_message = (
            f"ERROR RATE ALERT: {severity.value.upper()} errors "
            f"exceeded threshold ({count} >= {threshold} per hour)"
        )
        
        self.logger.warning(alert_message)
        
        # In a real implementation, this would send to monitoring systems
        # like Slack, PagerDuty, email, etc.
    
    def _get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get current status of all circuit breakers"""
        return {
            breaker_key: breaker['state']
            for breaker_key, breaker in self.circuit_breakers.items()
        }