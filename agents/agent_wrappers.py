"""
HAILEI Agent Wrapper System

Wraps existing CrewAI agents for integration with the conversational orchestrator.
Provides unified interface for agent execution with conversation context.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import threading
import json

from crewai import Agent, Task
from crewai.agent import Agent as CrewAIAgent


@dataclass
class AgentExecutionResult:
    """Result of agent execution with metadata"""
    agent_id: str
    task_description: str
    output: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'task_description': self.task_description,
            'output': self.output,
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class ConversationalAgentWrapper:
    """
    Wrapper for CrewAI agents to enable conversational orchestration.
    
    Features:
    - Context injection for conversation awareness
    - Async execution support
    - Error handling and retry logic
    - Memory management for agent state
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_id: str,
        crewai_agent: Agent,
        max_retries: int = 2,
        timeout_seconds: int = 300,
        enable_logging: bool = True
    ):
        """
        Initialize agent wrapper.
        
        Args:
            agent_id: Unique identifier for the agent
            crewai_agent: CrewAI Agent instance
            max_retries: Maximum retry attempts for failed executions
            timeout_seconds: Timeout for agent execution
            enable_logging: Enable detailed logging
        """
        self.agent_id = agent_id
        self.crewai_agent = crewai_agent
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Agent state
        self.execution_count = 0
        self.last_execution_time = None
        self.conversation_memory: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0,
            'successful_execution_time': 0.0
        }
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger(f'hailei_agent_wrapper_{agent_id}')
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
    
    async def execute_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AgentExecutionResult:
        """
        Execute task with conversational context.
        
        Args:
            task_description: Task for the agent to perform
            context: Additional context for execution
            conversation_history: Recent conversation history
            
        Returns:
            AgentExecutionResult: Execution result with metadata
        """
        start_time = datetime.now()
        self.execution_count += 1
        
        self.logger.info(f"Executing task for agent {self.agent_id}: {task_description[:100]}...")
        
        # Prepare enhanced task with context
        enhanced_task = self._prepare_contextual_task(
            task_description, context, conversation_history
        )
        
        # Execute with retries
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_with_timeout(enhanced_task)
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                self.last_execution_time = execution_time
                
                # Update performance metrics
                self._update_performance_metrics(execution_time, True)
                
                # Update conversation memory
                self._update_conversation_memory(task_description, result, context)
                
                success_result = AgentExecutionResult(
                    agent_id=self.agent_id,
                    task_description=task_description,
                    output=result,
                    execution_time=execution_time,
                    success=True,
                    metadata={
                        'attempt': attempt + 1,
                        'context_provided': context is not None,
                        'conversation_history_length': len(conversation_history) if conversation_history else 0,
                        'agent_role': self.crewai_agent.role,
                        'execution_timestamp': start_time.isoformat()
                    }
                )
                
                self.logger.info(f"Agent {self.agent_id} completed task successfully in {execution_time:.2f}s")
                return success_result
                
            except Exception as e:
                self.logger.warning(f"Agent {self.agent_id} execution attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    # Final failure
                    execution_time = (datetime.now() - start_time).total_seconds()
                    self._update_performance_metrics(execution_time, False)
                    
                    error_result = AgentExecutionResult(
                        agent_id=self.agent_id,
                        task_description=task_description,
                        output="",
                        execution_time=execution_time,
                        success=False,
                        error_message=str(e),
                        metadata={
                            'attempts': self.max_retries + 1,
                            'final_error': str(e),
                            'agent_role': self.crewai_agent.role,
                            'execution_timestamp': start_time.isoformat()
                        }
                    )
                    
                    self.logger.error(f"Agent {self.agent_id} failed after {self.max_retries + 1} attempts")
                    return error_result
                
                # Wait before retry
                await asyncio.sleep(1.0 * (attempt + 1))
    
    async def _execute_with_timeout(self, task: Task) -> str:
        """Execute task with timeout handling"""
        try:
            # Run the task execution in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_crewai_task, task),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Task execution timed out after {self.timeout_seconds} seconds")
    
    def _execute_crewai_task(self, task: Task) -> str:
        """Execute CrewAI task synchronously"""
        try:
            # Execute the task with the agent
            result = self.crewai_agent.execute_task(task)
            
            # Extract result string
            if hasattr(result, 'raw'):
                return str(result.raw)
            elif hasattr(result, 'output'):
                return str(result.output)
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"CrewAI task execution error: {e}")
            raise
    
    def _prepare_contextual_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> Task:
        """Prepare task with conversational context"""
        
        # Build enhanced task description
        enhanced_description = task_description
        
        # Add context if provided
        if context:
            context_section = self._format_context_section(context)
            enhanced_description += f"\\n\\n{context_section}"
        
        # Add conversation history if provided
        if conversation_history:
            history_section = self._format_conversation_history(conversation_history)
            enhanced_description += f"\\n\\n{history_section}"
        
        # Add agent memory if available
        if self.conversation_memory:
            memory_section = self._format_agent_memory()
            enhanced_description += f"\\n\\n{memory_section}"
        
        # Create CrewAI Task
        task = Task(
            description=enhanced_description,
            agent=self.crewai_agent,
            expected_output="Comprehensive response that addresses the task requirements while maintaining conversation context and building upon previous work."
        )
        
        return task
    
    def _format_context_section(self, context: Dict[str, Any]) -> str:
        """Format context information for task description"""
        formatted_sections = []
        
        # Course request information
        if 'course_request' in context:
            course_req = context['course_request']
            formatted_sections.append(f"""
COURSE CONTEXT:
- Title: {course_req.get('course_title', 'N/A')}
- Level: {course_req.get('course_level', 'N/A')}
- Duration: {course_req.get('course_duration_weeks', 'N/A')} weeks
- Credits: {course_req.get('course_credits', 'N/A')}
- Description: {course_req.get('course_description', 'N/A')[:200]}...
""")
        
        # Framework specifications
        if 'frameworks' in context:
            frameworks = context['frameworks']
            if 'kdka' in frameworks:
                formatted_sections.append("KDKA FRAMEWORK: Available for integration")
            if 'prrr' in frameworks:
                formatted_sections.append("PRRR FRAMEWORK: Available for integration")
        
        # Previous outputs from other agents
        if 'previous_outputs' in context:
            prev_outputs = context['previous_outputs']
            if prev_outputs:
                formatted_sections.append("PREVIOUS PHASE OUTPUTS:")
                for agent_id, output_info in prev_outputs.items():
                    formatted_sections.append(f"- {agent_id} ({output_info.get('phase', 'unknown')}): {output_info.get('content', '')[:150]}...")
        
        # Current phase information
        if 'current_phase' in context:
            formatted_sections.append(f"CURRENT PHASE: {context['current_phase']}")
        
        return "\\n".join(formatted_sections)
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context"""
        if not conversation_history:
            return ""
        
        history_lines = ["RECENT CONVERSATION:"]
        for turn in conversation_history[-5:]:  # Last 5 turns
            speaker = turn.get('speaker', 'unknown')
            message = turn.get('message', '')[:100]
            history_lines.append(f"- {speaker}: {message}...")
        
        return "\\n".join(history_lines)
    
    def _format_agent_memory(self) -> str:
        """Format agent memory for context"""
        if not self.conversation_memory:
            return ""
        
        memory_lines = ["AGENT MEMORY:"]
        for key, value in self.conversation_memory.items():
            if isinstance(value, str):
                memory_lines.append(f"- {key}: {value[:100]}...")
            else:
                memory_lines.append(f"- {key}: {str(value)[:100]}...")
        
        return "\\n".join(memory_lines)
    
    def _update_conversation_memory(
        self,
        task_description: str,
        result: str,
        context: Optional[Dict[str, Any]]
    ):
        """Update agent's conversation memory"""
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'task': task_description[:100],
            'result_summary': result[:200],
            'context_phase': context.get('current_phase') if context else None
        }
        
        # Store recent executions (limit to last 5)
        if 'recent_executions' not in self.conversation_memory:
            self.conversation_memory['recent_executions'] = []
        
        self.conversation_memory['recent_executions'].append(memory_entry)
        
        # Keep only recent executions
        if len(self.conversation_memory['recent_executions']) > 5:
            self.conversation_memory['recent_executions'] = self.conversation_memory['recent_executions'][-5:]
        
        # Update execution count
        self.conversation_memory['total_executions'] = self.execution_count
        self.conversation_memory['last_execution'] = datetime.now().isoformat()
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update agent performance metrics"""
        metrics = self.performance_metrics
        
        metrics['total_executions'] += 1
        metrics['total_execution_time'] += execution_time
        
        if success:
            metrics['successful_executions'] += 1
            metrics['successful_execution_time'] += execution_time
        else:
            metrics['failed_executions'] += 1
        
        # Calculate average execution time (only successful executions)
        if metrics['successful_executions'] > 0:
            metrics['average_execution_time'] = (
                metrics['successful_execution_time'] / metrics['successful_executions']
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            **self.performance_metrics,
            'success_rate': (
                self.performance_metrics['successful_executions'] / 
                max(self.performance_metrics['total_executions'], 1)
            ),
            'last_execution_time': self.last_execution_time,
            'agent_id': self.agent_id,
            'agent_role': self.crewai_agent.role
        }
    
    def get_conversation_memory(self) -> Dict[str, Any]:
        """Get agent conversation memory"""
        return self.conversation_memory.copy()
    
    def update_memory(self, key: str, value: Any):
        """Update specific memory entry"""
        self.conversation_memory[key] = value
        self.logger.info(f"Updated memory for agent {self.agent_id}: {key}")
    
    def clear_memory(self):
        """Clear agent conversation memory"""
        self.conversation_memory.clear()
        self.logger.info(f"Cleared memory for agent {self.agent_id}")
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0,
            'successful_execution_time': 0.0
        }
        self.execution_count = 0
        self.last_execution_time = None
        self.logger.info(f"Reset performance metrics for agent {self.agent_id}")


class AgentExecutor:
    """
    Manages execution of multiple conversational agent wrappers.
    
    Features:
    - Parallel agent execution
    - Load balancing and resource management
    - Agent health monitoring
    - Execution queue management
    """
    
    def __init__(
        self,
        max_concurrent_executions: int = 3,
        enable_logging: bool = True
    ):
        """
        Initialize agent executor.
        
        Args:
            max_concurrent_executions: Maximum concurrent agent executions
            enable_logging: Enable detailed logging
        """
        self.max_concurrent_executions = max_concurrent_executions
        
        # Agent management
        self.agents: Dict[str, ConversationalAgentWrapper] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger('hailei_agent_executor')
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
    
    def register_agent(self, agent_wrapper: ConversationalAgentWrapper):
        """Register agent wrapper for execution"""
        self.agents[agent_wrapper.agent_id] = agent_wrapper
        self.logger.info(f"Registered agent: {agent_wrapper.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent wrapper"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def execute_agent_task(
        self,
        agent_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AgentExecutionResult:
        """
        Execute task with specific agent.
        
        Args:
            agent_id: Agent identifier
            task_description: Task description
            context: Execution context
            conversation_history: Conversation history
            
        Returns:
            AgentExecutionResult: Execution result
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent_wrapper = self.agents[agent_id]
        
        # Use semaphore to limit concurrent executions
        async with self.execution_semaphore:
            self.logger.info(f"Starting execution for agent {agent_id}")
            
            try:
                result = await agent_wrapper.execute_task(
                    task_description, context, conversation_history
                )
                
                self.logger.info(f"Completed execution for agent {agent_id}")
                return result
                
            except Exception as e:
                self.logger.error(f"Execution failed for agent {agent_id}: {e}")
                raise
    
    async def execute_multiple_agents(
        self,
        agent_tasks: List[Dict[str, Any]]
    ) -> Dict[str, AgentExecutionResult]:
        """
        Execute multiple agents in parallel.
        
        Args:
            agent_tasks: List of agent task specifications
            
        Returns:
            Dict mapping agent_id to execution results
        """
        self.logger.info(f"Starting parallel execution for {len(agent_tasks)} agents")
        
        # Create execution tasks
        execution_tasks = {}
        for task_spec in agent_tasks:
            agent_id = task_spec['agent_id']
            task_description = task_spec['task_description']
            context = task_spec.get('context')
            conversation_history = task_spec.get('conversation_history')
            
            execution_tasks[agent_id] = asyncio.create_task(
                self.execute_agent_task(agent_id, task_description, context, conversation_history)
            )
        
        # Wait for all executions to complete
        results = {}
        for agent_id, task in execution_tasks.items():
            try:
                results[agent_id] = await task
            except Exception as e:
                self.logger.error(f"Failed execution for agent {agent_id}: {e}")
                results[agent_id] = AgentExecutionResult(
                    agent_id=agent_id,
                    task_description="Failed to execute",
                    output="",
                    execution_time=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        self.logger.info(f"Completed parallel execution for {len(agent_tasks)} agents")
        return results
    
    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents"""
        if agent_id:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                return {
                    'agent_id': agent_id,
                    'performance_metrics': agent.get_performance_metrics(),
                    'memory_size': len(agent.get_conversation_memory()),
                    'is_active': agent_id in self.active_executions
                }
            else:
                return {}
        else:
            return {
                agent_id: {
                    'agent_id': agent_id,
                    'performance_metrics': agent.get_performance_metrics(),
                    'memory_size': len(agent.get_conversation_memory()),
                    'is_active': agent_id in self.active_executions
                }
                for agent_id, agent in self.agents.items()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'registered_agents': list(self.agents.keys()),
            'max_concurrent_executions': self.max_concurrent_executions,
            'active_executions': len(self.active_executions),
            'available_slots': self.execution_semaphore._value,
            'total_agents': len(self.agents)
        }