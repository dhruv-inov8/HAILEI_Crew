"""
HAILEI Parallel Orchestration System

Advanced parallel execution engine for coordinating multiple agents simultaneously.
Intelligently determines when agents can work in parallel vs sequential execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading

from .conversation_state import ConversationState, AgentOutput, PhaseStatus
from agents.agent_wrappers import AgentExecutionResult


class ExecutionMode(Enum):
    """Execution modes for agent coordination"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class AgentDependency(Enum):
    """Types of dependencies between agents"""
    BLOCKS = "blocks"  # Must complete before next can start
    INFORMS = "informs"  # Output enhances next agent but not required
    INDEPENDENT = "independent"  # No dependency
    COLLABORATIVE = "collaborative"  # Can work together simultaneously


@dataclass
class ParallelTask:
    """Represents a task that can be executed in parallel"""
    agent_id: str
    task_description: str
    context: Dict[str, Any]
    dependencies: List[str]  # Agent IDs this task depends on
    priority: int = 1  # Higher number = higher priority
    estimated_duration: float = 60.0  # Estimated seconds
    can_be_interrupted: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'task_description': self.task_description,
            'dependencies': self.dependencies,
            'priority': self.priority,
            'estimated_duration': self.estimated_duration,
            'can_be_interrupted': self.can_be_interrupted
        }


@dataclass
class ExecutionPlan:
    """Plan for executing multiple agents"""
    execution_mode: ExecutionMode
    execution_groups: List[List[str]]  # Groups of agents that can run in parallel
    total_estimated_time: float
    parallel_efficiency: float  # 0-1, efficiency gain from parallelization
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_mode': self.execution_mode.value,
            'execution_groups': self.execution_groups,
            'total_estimated_time': self.total_estimated_time,
            'parallel_efficiency': self.parallel_efficiency
        }


class ParallelOrchestrator:
    """
    Advanced parallel execution system for HAILEI agents.
    
    Features:
    - Intelligent dependency analysis
    - Dynamic execution planning
    - Resource-aware scheduling
    - Real-time coordination
    - Failure handling and recovery
    """
    
    def __init__(
        self,
        main_orchestrator,
        max_parallel_agents: int = 3,
        enable_logging: bool = True
    ):
        """Initialize parallel orchestrator"""
        self.main_orchestrator = main_orchestrator
        self.max_parallel_agents = max_parallel_agents
        
        # Execution state
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_results: Dict[str, AgentExecutionResult] = {}
        self.execution_lock = threading.Lock()
        
        # Dependency configuration
        self.agent_dependencies = self._initialize_agent_dependencies()
        self.phase_execution_modes = self._initialize_phase_modes()
        
        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.parallel_efficiency_stats: Dict[str, float] = {}
        
        # Logging
        if enable_logging:
            self.logger = logging.getLogger('hailei_parallel_orchestrator')
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
    
    def _initialize_agent_dependencies(self) -> Dict[str, Dict[str, AgentDependency]]:
        """Initialize agent dependency matrix"""
        return {
            # IPDAi dependencies
            'ipdai_agent': {
                'hailei4t_coordinator_agent': AgentDependency.INFORMS,
                'cauthai_agent': AgentDependency.BLOCKS,  # CAuthAi needs IPDAi foundation
                'tfdai_agent': AgentDependency.INFORMS,
                'editorai_agent': AgentDependency.INFORMS,
                'ethosai_agent': AgentDependency.INDEPENDENT,
                'searchai_agent': AgentDependency.INDEPENDENT
            },
            
            # CAuthAi dependencies  
            'cauthai_agent': {
                'ipdai_agent': AgentDependency.BLOCKS,  # Needs foundation first
                'tfdai_agent': AgentDependency.COLLABORATIVE,  # Can work together on content
                'editorai_agent': AgentDependency.BLOCKS,  # Editor needs content first
                'ethosai_agent': AgentDependency.INDEPENDENT,
                'searchai_agent': AgentDependency.INFORMS
            },
            
            # TFDAi dependencies
            'tfdai_agent': {
                'ipdai_agent': AgentDependency.INFORMS,
                'cauthai_agent': AgentDependency.COLLABORATIVE,  # Can work with content
                'editorai_agent': AgentDependency.INFORMS,
                'ethosai_agent': AgentDependency.INDEPENDENT,
                'searchai_agent': AgentDependency.INFORMS
            },
            
            # EditorAi dependencies
            'editorai_agent': {
                'ipdai_agent': AgentDependency.INFORMS,
                'cauthai_agent': AgentDependency.BLOCKS,  # Needs content to edit
                'tfdai_agent': AgentDependency.INFORMS,
                'ethosai_agent': AgentDependency.INDEPENDENT,
                'searchai_agent': AgentDependency.INDEPENDENT
            },
            
            # EthosAi dependencies (can run independently for ethical review)
            'ethosai_agent': {
                'ipdai_agent': AgentDependency.INDEPENDENT,
                'cauthai_agent': AgentDependency.INDEPENDENT,
                'tfdai_agent': AgentDependency.INDEPENDENT,
                'editorai_agent': AgentDependency.INDEPENDENT,
                'searchai_agent': AgentDependency.INDEPENDENT
            },
            
            # SearchAi dependencies (can run independently for resource gathering)
            'searchai_agent': {
                'ipdai_agent': AgentDependency.INDEPENDENT,
                'cauthai_agent': AgentDependency.INFORMS,
                'tfdai_agent': AgentDependency.INFORMS,
                'editorai_agent': AgentDependency.INDEPENDENT,
                'ethosai_agent': AgentDependency.INDEPENDENT
            }
        }
    
    def _initialize_phase_modes(self) -> Dict[str, ExecutionMode]:
        """Initialize execution modes for different phases"""
        return {
            'course_overview': ExecutionMode.SEQUENTIAL,  # Foundation must be sequential
            'foundation_design': ExecutionMode.HYBRID,  # Some parallel, some sequential
            'content_creation': ExecutionMode.PARALLEL,  # Multiple agents can work together
            'technical_design': ExecutionMode.PARALLEL,  # Technical work can be parallel
            'quality_review': ExecutionMode.PARALLEL,  # Multiple reviewers can work simultaneously
            'ethical_audit': ExecutionMode.PARALLEL,  # Independent ethical review
            'final_integration': ExecutionMode.SEQUENTIAL  # Final integration must be coordinated
        }
    
    async def execute_phase_with_parallel_coordination(
        self,
        phase_id: str,
        agent_tasks: List[ParallelTask],
        conversation_state: ConversationState
    ) -> Dict[str, AgentOutput]:
        """
        Execute a phase with intelligent parallel coordination.
        
        Args:
            phase_id: Phase identifier
            agent_tasks: List of tasks to execute
            conversation_state: Current conversation state
            
        Returns:
            Dict of agent_id -> AgentOutput results
        """
        self.logger.info(f"Starting parallel coordination for phase: {phase_id}")
        
        # Create execution plan
        execution_plan = self._create_execution_plan(phase_id, agent_tasks)
        self.logger.info(f"Execution plan: {execution_plan.execution_mode.value} mode with {len(execution_plan.execution_groups)} groups")
        
        # Execute according to plan
        start_time = datetime.now()
        results = await self._execute_plan(execution_plan, agent_tasks, conversation_state)
        end_time = datetime.now()
        
        # Track performance
        execution_time = (end_time - start_time).total_seconds()
        self._track_execution_performance(phase_id, execution_plan, execution_time, results)
        
        self.logger.info(f"Phase {phase_id} completed in {execution_time:.2f}s with {len(results)} agents")
        return results
    
    def _create_execution_plan(
        self,
        phase_id: str,
        agent_tasks: List[ParallelTask]
    ) -> ExecutionPlan:
        """Create optimal execution plan for the given tasks"""
        
        # Get phase execution mode preference
        preferred_mode = self.phase_execution_modes.get(phase_id, ExecutionMode.SEQUENTIAL)
        
        # Analyze dependencies to determine actual execution groups
        execution_groups = self._analyze_dependencies(agent_tasks)
        
        # Calculate efficiency metrics
        total_sequential_time = sum(task.estimated_duration for task in agent_tasks)
        total_parallel_time = self._calculate_parallel_time(execution_groups, agent_tasks)
        efficiency = max(0, (total_sequential_time - total_parallel_time) / total_sequential_time)
        
        # Determine final execution mode
        if preferred_mode == ExecutionMode.SEQUENTIAL or len(execution_groups) == len(agent_tasks):
            final_mode = ExecutionMode.SEQUENTIAL
            execution_groups = [[task.agent_id] for task in agent_tasks]
        elif len(execution_groups) == 1:
            final_mode = ExecutionMode.PARALLEL
        else:
            final_mode = ExecutionMode.HYBRID
        
        return ExecutionPlan(
            execution_mode=final_mode,
            execution_groups=execution_groups,
            total_estimated_time=total_parallel_time,
            parallel_efficiency=efficiency
        )
    
    def _analyze_dependencies(self, agent_tasks: List[ParallelTask]) -> List[List[str]]:
        """Analyze task dependencies to create execution groups"""
        
        # Create dependency graph
        task_map = {task.agent_id: task for task in agent_tasks}
        agent_ids = [task.agent_id for task in agent_tasks]
        
        # Find agents that can run in parallel
        execution_groups = []
        remaining_agents = set(agent_ids)
        
        while remaining_agents:
            # Find agents with no unresolved dependencies
            ready_agents = []
            
            for agent_id in remaining_agents:
                task = task_map[agent_id]
                
                # Check if all dependencies are satisfied
                dependencies_satisfied = True
                for dep_agent in task.dependencies:
                    if dep_agent in remaining_agents:
                        # Check dependency type
                        if agent_id in self.agent_dependencies:
                            dep_type = self.agent_dependencies[agent_id].get(
                                dep_agent, AgentDependency.INDEPENDENT
                            )
                            if dep_type == AgentDependency.BLOCKS:
                                dependencies_satisfied = False
                                break
                
                if dependencies_satisfied:
                    ready_agents.append(agent_id)
            
            if not ready_agents:
                # Break circular dependencies by taking highest priority
                ready_agents = [max(remaining_agents, 
                                  key=lambda x: task_map[x].priority)]
            
            # Group collaborative agents together
            collaborative_group = []
            for agent_id in ready_agents:
                if not collaborative_group:
                    collaborative_group.append(agent_id)
                else:
                    # Check if this agent can collaborate with group
                    can_collaborate = True
                    for group_agent in collaborative_group:
                        if agent_id in self.agent_dependencies:
                            dep_type = self.agent_dependencies[agent_id].get(
                                group_agent, AgentDependency.INDEPENDENT
                            )
                            if dep_type not in [AgentDependency.COLLABORATIVE, AgentDependency.INDEPENDENT]:
                                can_collaborate = False
                                break
                    
                    if can_collaborate and len(collaborative_group) < self.max_parallel_agents:
                        collaborative_group.append(agent_id)
            
            execution_groups.append(collaborative_group)
            remaining_agents -= set(collaborative_group)
        
        return execution_groups
    
    def _calculate_parallel_time(
        self,
        execution_groups: List[List[str]],
        agent_tasks: List[ParallelTask]
    ) -> float:
        """Calculate total time for parallel execution"""
        task_map = {task.agent_id: task for task in agent_tasks}
        total_time = 0.0
        
        for group in execution_groups:
            # Time for group is maximum time of agents in group
            group_time = max(task_map[agent_id].estimated_duration for agent_id in group)
            total_time += group_time
        
        return total_time
    
    async def _execute_plan(
        self,
        execution_plan: ExecutionPlan,
        agent_tasks: List[ParallelTask],
        conversation_state: ConversationState
    ) -> Dict[str, AgentOutput]:
        """Execute the agents according to the execution plan"""
        
        task_map = {task.agent_id: task for task in agent_tasks}
        all_results = {}
        
        for group_index, agent_group in enumerate(execution_plan.execution_groups):
            self.logger.info(f"Executing group {group_index + 1}/{len(execution_plan.execution_groups)}: {agent_group}")
            
            if len(agent_group) == 1:
                # Sequential execution
                agent_id = agent_group[0]
                task = task_map[agent_id]
                result = await self._execute_single_agent(agent_id, task, conversation_state)
                all_results[agent_id] = result
                
                # Update conversation state with result
                conversation_state.add_agent_output(result)
                
            else:
                # Parallel execution
                group_results = await self._execute_agent_group(agent_group, task_map, conversation_state)
                all_results.update(group_results)
                
                # Update conversation state with all results
                for result in group_results.values():
                    conversation_state.add_agent_output(result)
        
        return all_results
    
    async def _execute_single_agent(
        self,
        agent_id: str,
        task: ParallelTask,
        conversation_state: ConversationState
    ) -> AgentOutput:
        """Execute a single agent task"""
        
        self.logger.info(f"Executing single agent: {agent_id}")
        
        # Use main orchestrator to execute agent
        try:
            result = await self.main_orchestrator.activate_agent(
                agent_id=agent_id,
                task_description=task.task_description,
                context=task.context
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing agent {agent_id}: {e}")
            # Create error result
            from .conversation_state import AgentOutput
            return AgentOutput(
                agent_id=agent_id,
                agent_name=f"Agent {agent_id}",
                phase=conversation_state.current_phase or "unknown",
                content=f"Agent execution failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={'error': True, 'error_message': str(e)}
            )
    
    async def _execute_agent_group(
        self,
        agent_group: List[str],
        task_map: Dict[str, ParallelTask],
        conversation_state: ConversationState
    ) -> Dict[str, AgentOutput]:
        """Execute a group of agents in parallel"""
        
        self.logger.info(f"Executing parallel group: {agent_group}")
        
        # Create tasks for parallel execution
        tasks = []
        for agent_id in agent_group:
            task = task_map[agent_id]
            agent_task = asyncio.create_task(
                self._execute_single_agent(agent_id, task, conversation_state)
            )
            tasks.append((agent_id, agent_task))
        
        # Wait for all tasks to complete
        results = {}
        for agent_id, agent_task in tasks:
            try:
                result = await agent_task
                results[agent_id] = result
            except Exception as e:
                self.logger.error(f"Parallel execution failed for {agent_id}: {e}")
                # Create error result
                from .conversation_state import AgentOutput
                results[agent_id] = AgentOutput(
                    agent_id=agent_id,
                    agent_name=f"Agent {agent_id}",
                    phase=conversation_state.current_phase or "unknown",
                    content=f"Parallel execution failed: {str(e)}",
                    timestamp=datetime.now(),
                    metadata={'error': True, 'error_message': str(e)}
                )
        
        return results
    
    def _track_execution_performance(
        self,
        phase_id: str,
        execution_plan: ExecutionPlan,
        actual_time: float,
        results: Dict[str, AgentOutput]
    ):
        """Track execution performance for optimization"""
        
        performance_data = {
            'phase_id': phase_id,
            'execution_mode': execution_plan.execution_mode.value,
            'estimated_time': execution_plan.total_estimated_time,
            'actual_time': actual_time,
            'parallel_efficiency': execution_plan.parallel_efficiency,
            'successful_agents': len([r for r in results.values() if not r.metadata.get('error', False)]),
            'total_agents': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.execution_history.append(performance_data)
        
        # Update efficiency stats
        self.parallel_efficiency_stats[phase_id] = execution_plan.parallel_efficiency
        
        self.logger.info(f"Performance tracked: {actual_time:.2f}s actual vs {execution_plan.total_estimated_time:.2f}s estimated")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        total_executions = len(self.execution_history)
        avg_efficiency = sum(self.parallel_efficiency_stats.values()) / len(self.parallel_efficiency_stats) if self.parallel_efficiency_stats else 0
        
        mode_counts = {}
        for execution in self.execution_history:
            mode = execution['execution_mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        return {
            'total_executions': total_executions,
            'average_parallel_efficiency': avg_efficiency,
            'execution_mode_distribution': mode_counts,
            'phase_efficiencies': self.parallel_efficiency_stats,
            'recent_executions': self.execution_history[-5:]  # Last 5 executions
        }
    
    def optimize_agent_dependencies(self, phase_results: Dict[str, List[Dict[str, Any]]]):
        """Optimize agent dependencies based on execution results"""
        # This could use ML to optimize dependencies based on actual performance
        # For now, it's a placeholder for future enhancement
        self.logger.info("Dependency optimization not yet implemented")
        pass