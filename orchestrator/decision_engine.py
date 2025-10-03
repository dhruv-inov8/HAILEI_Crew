"""
HAILEI Dynamic Decision Engine

Intelligent decision-making system that dynamically determines execution strategies,
agent selection, and workflow optimization based on context and performance data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

from .parallel_orchestrator import ExecutionMode, AgentDependency, ParallelTask
from .conversation_state import ConversationState, AgentOutput, PhaseStatus


class DecisionType(Enum):
    """Types of decisions the engine can make"""
    AGENT_SELECTION = "agent_selection"
    EXECUTION_MODE = "execution_mode"
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    RESOURCE_ALLOCATION = "resource_allocation"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"


class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DecisionContext:
    """Context for decision making"""
    phase_id: str
    current_state: ConversationState
    available_agents: List[str]
    execution_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    user_preferences: Dict[str, Any]
    time_constraints: Optional[Dict[str, Any]] = None
    resource_constraints: Optional[Dict[str, Any]] = None


@dataclass
class Decision:
    """Represents a decision made by the engine"""
    decision_type: DecisionType
    decision_value: Any
    confidence: ConfidenceLevel
    reasoning: str
    alternatives: List[Any]
    metrics_used: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_type': self.decision_type.value,
            'decision_value': self.decision_value,
            'confidence': self.confidence.value,
            'reasoning': self.reasoning,
            'alternatives': self.alternatives,
            'metrics_used': self.metrics_used,
            'timestamp': self.timestamp.isoformat()
        }


class DynamicDecisionEngine:
    """
    Advanced decision engine for HAILEI orchestration.
    
    Features:
    - Context-aware agent selection
    - Dynamic execution mode optimization
    - Performance-based learning
    - Resource-aware planning
    - Adaptive workflow adjustment
    """
    
    def __init__(self, enable_logging: bool = True):
        """Initialize the decision engine"""
        self.enable_logging = enable_logging
        
        # Decision history for learning
        self.decision_history: List[Decision] = []
        self.performance_feedback: Dict[str, float] = {}
        
        # Agent performance tracking
        self.agent_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.agent_success_rates: Dict[str, float] = {}
        self.agent_average_times: Dict[str, float] = {}
        
        # Phase-specific optimization data
        self.phase_optimization_data: Dict[str, Dict[str, Any]] = {}
        
        # Decision weights and thresholds
        self.decision_weights = {
            'performance_weight': 0.4,
            'speed_weight': 0.3,
            'accuracy_weight': 0.2,
            'user_preference_weight': 0.1
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'low': 0.6,
            'medium': 0.75,
            'high': 0.9,
            'very_high': 0.95
        }
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger('hailei_decision_engine')
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
    
    async def make_execution_decision(
        self,
        context: DecisionContext
    ) -> Decision:
        """
        Make an intelligent decision about execution strategy.
        
        Args:
            context: Decision context with current state and constraints
            
        Returns:
            Decision object with recommended execution strategy
        """
        self.logger.info(f"Making execution decision for phase: {context.phase_id}")
        
        # Analyze current context
        analysis = self._analyze_context(context)
        
        # Calculate execution mode recommendation
        execution_mode = self._determine_optimal_execution_mode(context, analysis)
        
        # Build decision reasoning
        reasoning = self._build_execution_reasoning(execution_mode, analysis)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(analysis)
        
        # Generate alternatives
        alternatives = self._generate_execution_alternatives(context, execution_mode)
        
        decision = Decision(
            decision_type=DecisionType.EXECUTION_MODE,
            decision_value=execution_mode,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            metrics_used=analysis['metrics'],
            timestamp=datetime.now()
        )
        
        # Store decision for learning
        self.decision_history.append(decision)
        
        self.logger.info(f"Execution decision: {execution_mode.value} (confidence: {confidence.value})")
        return decision
    
    async def make_agent_selection_decision(
        self,
        context: DecisionContext,
        required_skills: List[str]
    ) -> Decision:
        """
        Make intelligent agent selection based on performance and context.
        
        Args:
            context: Decision context
            required_skills: Skills needed for the task
            
        Returns:
            Decision with recommended agent selection
        """
        self.logger.info(f"Making agent selection decision for skills: {required_skills}")
        
        # Score available agents
        agent_scores = self._score_agents(context, required_skills)
        
        # Select best agent
        best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        
        # Build reasoning
        reasoning = self._build_agent_selection_reasoning(agent_scores, best_agent)
        
        # Calculate confidence
        confidence = self._calculate_agent_selection_confidence(agent_scores)
        
        # Generate alternatives
        alternatives = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
        alternatives = [agent for agent, score in alternatives]
        
        decision = Decision(
            decision_type=DecisionType.AGENT_SELECTION,
            decision_value=best_agent,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            metrics_used=agent_scores,
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        
        self.logger.info(f"Agent selection: {best_agent} (confidence: {confidence.value})")
        return decision
    
    async def optimize_workflow_priorities(
        self,
        context: DecisionContext,
        tasks: List[ParallelTask]
    ) -> Decision:
        """
        Optimize task priorities based on context and performance data.
        
        Args:
            context: Decision context
            tasks: List of tasks to prioritize
            
        Returns:
            Decision with optimized priority assignments
        """
        self.logger.info(f"Optimizing workflow priorities for {len(tasks)} tasks")
        
        # Calculate optimization metrics
        priority_analysis = self._analyze_priority_optimization(context, tasks)
        
        # Generate optimized priorities
        optimized_priorities = self._optimize_task_priorities(tasks, priority_analysis)
        
        # Build reasoning
        reasoning = self._build_priority_reasoning(priority_analysis, optimized_priorities)
        
        # Calculate confidence
        confidence = self._calculate_priority_confidence(priority_analysis)
        
        decision = Decision(
            decision_type=DecisionType.PRIORITY_ADJUSTMENT,
            decision_value=optimized_priorities,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=[],
            metrics_used=priority_analysis['metrics'],
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        
        self.logger.info(f"Priority optimization complete (confidence: {confidence.value})")
        return decision
    
    def _analyze_context(self, context: DecisionContext) -> Dict[str, Any]:
        """Analyze decision context to extract relevant metrics"""
        analysis = {
            'phase_complexity': self._calculate_phase_complexity(context.phase_id),
            'agent_availability': len(context.available_agents),
            'historical_performance': self._get_historical_performance(context.phase_id),
            'time_pressure': self._assess_time_pressure(context.time_constraints),
            'resource_availability': self._assess_resource_availability(context.resource_constraints),
            'user_preferences': context.user_preferences,
            'metrics': {}
        }
        
        # Calculate composite metrics
        analysis['metrics'] = {
            'complexity_score': analysis['phase_complexity'],
            'performance_score': analysis['historical_performance'],
            'urgency_score': analysis['time_pressure'],
            'resource_score': analysis['resource_availability']
        }
        
        return analysis
    
    def _determine_optimal_execution_mode(
        self,
        context: DecisionContext,
        analysis: Dict[str, Any]
    ) -> ExecutionMode:
        """Determine optimal execution mode based on analysis"""
        
        # Calculate mode scores
        mode_scores = {}
        
        # Sequential mode scoring
        sequential_score = (
            analysis['metrics']['complexity_score'] * 0.3 +
            analysis['metrics']['performance_score'] * 0.4 +
            (1.0 - analysis['metrics']['urgency_score']) * 0.3
        )
        mode_scores[ExecutionMode.SEQUENTIAL] = sequential_score
        
        # Parallel mode scoring
        parallel_score = (
            (1.0 - analysis['metrics']['complexity_score']) * 0.2 +
            analysis['metrics']['resource_score'] * 0.3 +
            analysis['metrics']['urgency_score'] * 0.3 +
            min(1.0, analysis['agent_availability'] / 3.0) * 0.2
        )
        mode_scores[ExecutionMode.PARALLEL] = parallel_score
        
        # Hybrid mode scoring
        hybrid_score = (
            analysis['metrics']['complexity_score'] * 0.25 +
            analysis['metrics']['performance_score'] * 0.25 +
            analysis['metrics']['resource_score'] * 0.25 +
            analysis['metrics']['urgency_score'] * 0.25
        )
        mode_scores[ExecutionMode.HYBRID] = hybrid_score
        
        # Select best mode
        best_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
        
        # Apply user preferences
        if 'execution_mode_preference' in context.user_preferences:
            preferred_mode = context.user_preferences['execution_mode_preference']
            if preferred_mode in mode_scores:
                # Boost preferred mode score
                mode_scores[ExecutionMode(preferred_mode)] *= 1.2
                best_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
        
        return best_mode
    
    def _score_agents(
        self,
        context: DecisionContext,
        required_skills: List[str]
    ) -> Dict[str, float]:
        """Score available agents based on performance and suitability"""
        
        agent_scores = {}
        
        for agent_id in context.available_agents:
            score = 0.0
            
            # Performance-based scoring
            if agent_id in self.agent_success_rates:
                score += self.agent_success_rates[agent_id] * self.decision_weights['performance_weight']
            else:
                score += 0.5 * self.decision_weights['performance_weight']  # Default for new agents
            
            # Speed-based scoring
            if agent_id in self.agent_average_times:
                avg_time = self.agent_average_times[agent_id]
                max_time = max(self.agent_average_times.values()) if self.agent_average_times else 100.0
                speed_score = 1.0 - (avg_time / max_time)
                score += speed_score * self.decision_weights['speed_weight']
            else:
                score += 0.5 * self.decision_weights['speed_weight']
            
            # Skill matching (simplified - would use actual skill data in production)
            skill_match_score = self._calculate_skill_match(agent_id, required_skills)
            score += skill_match_score * self.decision_weights['accuracy_weight']
            
            # User preference bonus
            if 'preferred_agents' in context.user_preferences:
                if agent_id in context.user_preferences['preferred_agents']:
                    score += self.decision_weights['user_preference_weight']
            
            agent_scores[agent_id] = score
        
        return agent_scores
    
    def _calculate_skill_match(self, agent_id: str, required_skills: List[str]) -> float:
        """Calculate how well an agent matches required skills"""
        # Simplified skill matching - in production this would use actual agent capabilities
        agent_skill_map = {
            'ipdai_agent': ['instructional_design', 'learning_objectives', 'assessment'],
            'cauthai_agent': ['content_creation', 'engagement', 'storytelling'],
            'tfdai_agent': ['technical_implementation', 'lms_integration', 'systems'],
            'editorai_agent': ['quality_review', 'editing', 'proofreading'],
            'ethosai_agent': ['ethical_review', 'compliance', 'accessibility'],
            'searchai_agent': ['research', 'resource_gathering', 'analysis']
        }
        
        agent_skills = agent_skill_map.get(agent_id, [])
        if not required_skills or not agent_skills:
            return 0.5  # Default match score
        
        matches = len(set(agent_skills) & set(required_skills))
        return matches / len(required_skills)
    
    def _calculate_phase_complexity(self, phase_id: str) -> float:
        """Calculate complexity score for a phase"""
        complexity_map = {
            'course_overview': 0.3,
            'foundation_design': 0.8,
            'content_creation': 0.9,
            'technical_design': 0.7,
            'quality_review': 0.6,
            'ethical_audit': 0.5,
            'final_integration': 0.8
        }
        return complexity_map.get(phase_id, 0.5)
    
    def _get_historical_performance(self, phase_id: str) -> float:
        """Get historical performance score for a phase"""
        if phase_id in self.phase_optimization_data:
            return self.phase_optimization_data[phase_id].get('success_rate', 0.5)
        return 0.5  # Default for new phases
    
    def _assess_time_pressure(self, time_constraints: Optional[Dict[str, Any]]) -> float:
        """Assess time pressure based on constraints"""
        if not time_constraints:
            return 0.3  # Low pressure if no constraints
        
        if 'deadline' in time_constraints:
            deadline = datetime.fromisoformat(time_constraints['deadline'])
            time_remaining = (deadline - datetime.now()).total_seconds()
            
            # Normalize to 0-1 scale (assuming 1 week = low pressure)
            week_seconds = 7 * 24 * 3600
            pressure = max(0, 1.0 - (time_remaining / week_seconds))
            return min(1.0, pressure)
        
        return 0.3
    
    def _assess_resource_availability(self, resource_constraints: Optional[Dict[str, Any]]) -> float:
        """Assess resource availability"""
        if not resource_constraints:
            return 1.0  # Full resources available
        
        # Calculate resource score based on constraints
        available_resources = resource_constraints.get('available_compute', 1.0)
        max_resources = resource_constraints.get('max_compute', 1.0)
        
        return available_resources / max_resources if max_resources > 0 else 1.0
    
    def _calculate_decision_confidence(self, analysis: Dict[str, Any]) -> ConfidenceLevel:
        """Calculate confidence level for a decision"""
        # Combine various factors to determine confidence
        confidence_score = (
            analysis['metrics']['performance_score'] * 0.3 +
            analysis['metrics']['complexity_score'] * 0.2 +
            analysis['metrics']['resource_score'] * 0.2 +
            (1.0 if analysis['agent_availability'] >= 2 else 0.5) * 0.3
        )
        
        if confidence_score >= self.confidence_thresholds['very_high']:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= self.confidence_thresholds['high']:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.confidence_thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_agent_selection_confidence(self, agent_scores: Dict[str, float]) -> ConfidenceLevel:
        """Calculate confidence for agent selection"""
        if not agent_scores:
            return ConfidenceLevel.LOW
        
        scores = list(agent_scores.values())
        max_score = max(scores)
        score_variance = np.var(scores) if len(scores) > 1 else 0
        
        # High confidence if clear winner and low variance
        if max_score > 0.8 and score_variance < 0.1:
            return ConfidenceLevel.VERY_HIGH
        elif max_score > 0.7 and score_variance < 0.2:
            return ConfidenceLevel.HIGH
        elif max_score > 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _build_execution_reasoning(
        self,
        execution_mode: ExecutionMode,
        analysis: Dict[str, Any]
    ) -> str:
        """Build human-readable reasoning for execution decision"""
        
        reasoning_parts = [
            f"Selected {execution_mode.value} execution mode based on:"
        ]
        
        if execution_mode == ExecutionMode.PARALLEL:
            reasoning_parts.extend([
                f"• High resource availability ({analysis['metrics']['resource_score']:.2f})",
                f"• Multiple agents available ({analysis['agent_availability']})",
                f"• Time pressure requires efficiency ({analysis['metrics']['urgency_score']:.2f})"
            ])
        elif execution_mode == ExecutionMode.SEQUENTIAL:
            reasoning_parts.extend([
                f"• High phase complexity ({analysis['metrics']['complexity_score']:.2f})",
                f"• Sequential dependencies require coordination",
                f"• Historical performance favors careful execution"
            ])
        else:  # HYBRID
            reasoning_parts.extend([
                f"• Balanced complexity and resource requirements",
                f"• Mixed dependency patterns in tasks",
                f"• Optimal efficiency through selective parallelization"
            ])
        
        return "\n".join(reasoning_parts)
    
    def _build_agent_selection_reasoning(
        self,
        agent_scores: Dict[str, float],
        selected_agent: str
    ) -> str:
        """Build reasoning for agent selection"""
        
        selected_score = agent_scores[selected_agent]
        
        reasoning = f"Selected {selected_agent} (score: {selected_score:.3f}) based on:\n"
        
        if selected_agent in self.agent_success_rates:
            reasoning += f"• Success rate: {self.agent_success_rates[selected_agent]:.1%}\n"
        
        if selected_agent in self.agent_average_times:
            reasoning += f"• Average execution time: {self.agent_average_times[selected_agent]:.1f}s\n"
        
        reasoning += "• Strong skill match for required capabilities"
        
        return reasoning
    
    def _generate_execution_alternatives(
        self,
        context: DecisionContext,
        selected_mode: ExecutionMode
    ) -> List[ExecutionMode]:
        """Generate alternative execution modes"""
        all_modes = [ExecutionMode.SEQUENTIAL, ExecutionMode.PARALLEL, ExecutionMode.HYBRID]
        alternatives = [mode for mode in all_modes if mode != selected_mode]
        return alternatives
    
    def _analyze_priority_optimization(
        self,
        context: DecisionContext,
        tasks: List[ParallelTask]
    ) -> Dict[str, Any]:
        """Analyze tasks for priority optimization"""
        
        return {
            'task_complexities': {task.agent_id: len(task.dependencies) for task in tasks},
            'agent_performance': {task.agent_id: self.agent_success_rates.get(task.agent_id, 0.5) for task in tasks},
            'time_estimates': {task.agent_id: task.estimated_duration for task in tasks},
            'current_priorities': {task.agent_id: task.priority for task in tasks},
            'metrics': {
                'total_tasks': len(tasks),
                'avg_complexity': sum(len(task.dependencies) for task in tasks) / len(tasks),
                'total_estimated_time': sum(task.estimated_duration for task in tasks)
            }
        }
    
    def _optimize_task_priorities(
        self,
        tasks: List[ParallelTask],
        analysis: Dict[str, Any]
    ) -> Dict[str, int]:
        """Optimize task priorities based on analysis"""
        
        optimized_priorities = {}
        
        for task in tasks:
            agent_id = task.agent_id
            
            # Calculate priority score
            performance_factor = analysis['agent_performance'][agent_id]
            complexity_factor = 1.0 / (analysis['task_complexities'][agent_id] + 1)
            time_factor = 1.0 / (analysis['time_estimates'][agent_id] / 60.0)  # Normalize to minutes
            
            priority_score = (
                performance_factor * 0.4 +
                complexity_factor * 0.3 +
                time_factor * 0.3
            )
            
            # Convert to integer priority (1-5 scale)
            optimized_priority = max(1, min(5, int(priority_score * 5)))
            optimized_priorities[agent_id] = optimized_priority
        
        return optimized_priorities
    
    def _build_priority_reasoning(
        self,
        analysis: Dict[str, Any],
        optimized_priorities: Dict[str, int]
    ) -> str:
        """Build reasoning for priority optimization"""
        
        reasoning = f"Optimized priorities for {analysis['metrics']['total_tasks']} tasks:\n"
        
        # Sort by optimized priority
        sorted_priorities = sorted(optimized_priorities.items(), key=lambda x: x[1], reverse=True)
        
        for agent_id, priority in sorted_priorities[:3]:  # Show top 3
            performance = analysis['agent_performance'][agent_id]
            complexity = analysis['task_complexities'][agent_id]
            reasoning += f"• {agent_id}: Priority {priority} (performance: {performance:.1%}, deps: {complexity})\n"
        
        reasoning += f"Based on agent performance, task complexity, and time estimates"
        
        return reasoning
    
    def _calculate_priority_confidence(self, analysis: Dict[str, Any]) -> ConfidenceLevel:
        """Calculate confidence for priority optimization"""
        
        # Higher confidence with more data and consistent performance
        performance_variance = np.var(list(analysis['agent_performance'].values()))
        data_completeness = len([p for p in analysis['agent_performance'].values() if p != 0.5]) / len(analysis['agent_performance'])
        
        confidence_score = (1.0 - performance_variance) * 0.5 + data_completeness * 0.5
        
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def update_performance_feedback(
        self,
        decision_id: str,
        performance_score: float,
        actual_outcome: Dict[str, Any]
    ):
        """Update decision engine with performance feedback for learning"""
        
        self.performance_feedback[decision_id] = performance_score
        
        # Update agent performance if applicable
        if 'agent_id' in actual_outcome:
            agent_id = actual_outcome['agent_id']
            success = actual_outcome.get('success', False)
            execution_time = actual_outcome.get('execution_time', 0)
            
            # Update success rate
            if agent_id not in self.agent_success_rates:
                self.agent_success_rates[agent_id] = 0.5
            
            current_rate = self.agent_success_rates[agent_id]
            new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            self.agent_success_rates[agent_id] = new_rate
            
            # Update average time
            if agent_id not in self.agent_average_times:
                self.agent_average_times[agent_id] = execution_time
            else:
                current_avg = self.agent_average_times[agent_id]
                new_avg = current_avg * 0.8 + execution_time * 0.2
                self.agent_average_times[agent_id] = new_avg
        
        self.logger.info(f"Updated performance feedback for decision: {performance_score:.3f}")
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        
        if not self.decision_history:
            return {'message': 'No decisions made yet'}
        
        decision_types = defaultdict(int)
        confidence_distribution = defaultdict(int)
        
        for decision in self.decision_history:
            decision_types[decision.decision_type.value] += 1
            confidence_distribution[decision.confidence.value] += 1
        
        return {
            'total_decisions': len(self.decision_history),
            'decision_types': dict(decision_types),
            'confidence_distribution': dict(confidence_distribution),
            'agent_performance': self.agent_success_rates,
            'recent_decisions': [d.to_dict() for d in self.decision_history[-5:]]
        }