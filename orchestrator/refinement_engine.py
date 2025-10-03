"""
HAILEI Refinement Engine

Handles iterative improvement of agent outputs based on user feedback.
Enables true conversational refinement for perfect course design results.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .conversation_state import AgentOutput, AgentStatus


class RefinementEngine:
    """
    Manages iterative refinement of agent outputs based on user feedback.
    
    Features:
    - Feedback analysis and task refinement
    - Version control for iterative improvements
    - Context preservation across refinement cycles
    - Quality improvement tracking
    - Convergence detection to prevent endless refinement
    """
    
    def __init__(self, orchestrator):
        """Initialize with reference to main orchestrator"""
        self.orchestrator = orchestrator
        self.logger = logging.getLogger('hailei_refinement_engine')
        
        # Refinement configuration
        self.max_refinement_cycles = 5
        self.improvement_threshold = 0.8  # For future quality scoring
        
        # Feedback processing patterns
        self.feedback_patterns = {
            'content_adjustment': [
                'add', 'include', 'expand', 'elaborate', 'detail',
                'remove', 'delete', 'reduce', 'simplify', 'shorten'
            ],
            'tone_adjustment': [
                'formal', 'informal', 'professional', 'casual', 'academic',
                'friendly', 'serious', 'engaging', 'conversational'
            ],
            'structure_adjustment': [
                'organize', 'structure', 'reorder', 'sequence', 'flow',
                'format', 'layout', 'arrange', 'group', 'categorize'
            ],
            'clarity_improvement': [
                'clarify', 'explain', 'clear', 'confusing', 'unclear',
                'ambiguous', 'vague', 'specific', 'precise', 'detailed'
            ],
            'framework_alignment': [
                'kdka', 'prrr', 'bloom', 'taxonomy', 'framework',
                'personal', 'relatable', 'relative', 'real-world',
                'knowledge', 'delivery', 'context', 'assessment'
            ]
        }
    
    async def refine_output(
        self,
        agent_id: str,
        feedback: str,
        phase_id: Optional[str] = None
    ) -> AgentOutput:
        """
        Refine agent output based on user feedback.
        
        Args:
            agent_id: Agent whose output to refine
            feedback: User feedback for improvement
            phase_id: Phase context for refinement
            
        Returns:
            AgentOutput: Refined output for user review
        """
        if not self.orchestrator.conversation_state:
            raise ValueError("No active conversation session")
        
        phase_id = phase_id or self.orchestrator.conversation_state.current_phase
        
        # Get current output to refine
        current_output = self.orchestrator.conversation_state.get_latest_output(agent_id, phase_id)
        if not current_output:
            raise ValueError(f"No output found for agent {agent_id} in phase {phase_id}")
        
        # Check refinement cycle limits
        cycle_key = f"{agent_id}_{phase_id}"
        current_cycles = self.orchestrator.conversation_state.refinement_cycles.get(cycle_key, 0)
        
        if current_cycles >= self.max_refinement_cycles:
            self.logger.warning(f"Max refinement cycles reached for {agent_id} in {phase_id}")
            raise ValueError("Maximum refinement cycles reached. Please approve current output or start fresh.")
        
        self.logger.info(f"Refining output for {agent_id} based on feedback: {feedback[:100]}...")
        
        # Analyze feedback to determine refinement strategy
        refinement_strategy = self._analyze_feedback(feedback)
        
        # Create refined task description
        refined_task = self._create_refinement_task(
            agent_id, current_output, feedback, refinement_strategy, phase_id
        )
        
        # Update agent status
        self.orchestrator.conversation_state.set_active_agent(agent_id, AgentStatus.WORKING)
        
        # Add refinement note to current output
        current_output.refinement_notes.append(f"Refinement cycle {current_cycles + 1}: {feedback}")
        
        # Execute refined task
        refined_result = await self._execute_refinement_task(agent_id, refined_task)
        
        # Create new output version
        refined_output = AgentOutput(
            agent_id=agent_id,
            agent_name=current_output.agent_name,
            phase=phase_id,
            content=refined_result,
            timestamp=datetime.now(),
            version=current_output.version + 1,
            refinement_notes=current_output.refinement_notes.copy(),
            user_feedback=current_output.user_feedback.copy(),
            metadata={
                'refinement_strategy': refinement_strategy,
                'original_version': current_output.version,
                'feedback_analyzed': feedback,
                'refinement_cycle': current_cycles + 1
            }
        )
        
        # Add to conversation state
        self.orchestrator.conversation_state.add_agent_output(refined_output)
        
        # Update agent status
        self.orchestrator.conversation_state.set_active_agent(agent_id, AgentStatus.WAITING_FOR_HUMAN)
        
        # Add to conversation history
        self.orchestrator.conversation_state.add_conversation_turn(
            f"{current_output.agent_name} (refined)",
            f"Refined output based on feedback: {feedback[:100]}..."
        )
        
        self.logger.info(f"Completed refinement cycle {current_cycles + 1} for {agent_id}")
        
        return refined_output
    
    def _analyze_feedback(self, feedback: str) -> Dict[str, Any]:
        """
        Analyze user feedback to determine refinement strategy.
        
        Args:
            feedback: User feedback text
            
        Returns:
            Dict: Refinement strategy with detected patterns and priorities
        """
        feedback_lower = feedback.lower()
        detected_patterns = {}
        
        # Detect feedback patterns
        for pattern_type, keywords in self.feedback_patterns.items():
            matches = [keyword for keyword in keywords if keyword in feedback_lower]
            if matches:
                detected_patterns[pattern_type] = matches
        
        # Determine primary refinement focus
        primary_focus = 'general_improvement'
        if detected_patterns:
            # Priority order for refinement focus
            focus_priority = [
                'framework_alignment',
                'structure_adjustment', 
                'clarity_improvement',
                'content_adjustment',
                'tone_adjustment'
            ]
            
            for focus in focus_priority:
                if focus in detected_patterns:
                    primary_focus = focus
                    break
        
        # Detect sentiment and urgency
        sentiment = self._analyze_feedback_sentiment(feedback)
        urgency = self._detect_feedback_urgency(feedback)
        
        strategy = {
            'primary_focus': primary_focus,
            'detected_patterns': detected_patterns,
            'sentiment': sentiment,
            'urgency': urgency,
            'requires_major_revision': urgency == 'high' or len(detected_patterns) > 2,
            'specific_keywords': self._extract_specific_keywords(feedback)
        }
        
        self.logger.info(f"Feedback analysis: {strategy}")
        return strategy
    
    def _analyze_feedback_sentiment(self, feedback: str) -> str:
        """Analyze sentiment of feedback (positive, negative, neutral)"""
        positive_indicators = ['good', 'great', 'excellent', 'like', 'love', 'perfect', 'nice']
        negative_indicators = ['bad', 'poor', 'terrible', 'hate', 'dislike', 'wrong', 'awful']
        
        feedback_lower = feedback.lower()
        positive_count = sum(1 for word in positive_indicators if word in feedback_lower)
        negative_count = sum(1 for word in negative_indicators if word in feedback_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _detect_feedback_urgency(self, feedback: str) -> str:
        """Detect urgency level of feedback (low, medium, high)"""
        high_urgency = ['completely', 'totally', 'entirely', 'must', 'need to', 'have to', 'critical']
        medium_urgency = ['should', 'would like', 'prefer', 'better', 'improve']
        
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in high_urgency):
            return 'high'
        elif any(word in feedback_lower for word in medium_urgency):
            return 'medium'
        else:
            return 'low'
    
    def _extract_specific_keywords(self, feedback: str) -> List[str]:
        """Extract specific keywords and phrases for targeted refinement"""
        # Split feedback into words and extract meaningful terms
        words = feedback.lower().split()
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'can', 'may', 'might', 'must'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return top keywords (limit to prevent overwhelming)
        return meaningful_words[:10]
    
    def _create_refinement_task(
        self,
        agent_id: str,
        current_output: AgentOutput,
        feedback: str,
        strategy: Dict[str, Any],
        phase_id: str
    ) -> str:
        """
        Create specific refinement task based on feedback analysis.
        
        Args:
            agent_id: Agent identifier
            current_output: Current output to refine
            feedback: User feedback
            strategy: Refinement strategy from analysis
            phase_id: Phase context
            
        Returns:
            str: Refined task description
        """
        # Get original task context
        original_task = self.orchestrator.phase_manager._get_agent_task_description(agent_id, phase_id)
        
        # Create refinement-specific instructions
        refinement_instructions = self._get_refinement_instructions(strategy)
        
        # Build comprehensive refinement task
        refinement_task = f"""
REFINEMENT TASK - Version {current_output.version + 1}

ORIGINAL TASK CONTEXT:
{original_task}

CURRENT OUTPUT TO REFINE:
{current_output.content}

USER FEEDBACK:
"{feedback}"

REFINEMENT STRATEGY:
Primary Focus: {strategy['primary_focus']}
Detected Patterns: {strategy['detected_patterns']}
Urgency Level: {strategy['urgency']}
Requires Major Revision: {strategy['requires_major_revision']}

SPECIFIC REFINEMENT INSTRUCTIONS:
{refinement_instructions}

REQUIREMENTS FOR REFINED OUTPUT:
1. Address ALL specific feedback points mentioned by the user
2. Maintain consistency with approved outputs from previous phases
3. Preserve KDKA and PRRR framework alignment
4. Ensure educational quality and professional standards
5. Build upon strengths from current output while addressing concerns
6. Provide clear improvements that directly respond to user feedback

CONVERSATION CONTEXT:
{self._get_refinement_context()}

Please provide a completely refined output that addresses the user's feedback while maintaining the overall quality and framework compliance.
"""
        
        return refinement_task
    
    def _get_refinement_instructions(self, strategy: Dict[str, Any]) -> str:
        """Generate specific refinement instructions based on strategy"""
        instructions = []
        
        primary_focus = strategy['primary_focus']
        
        if primary_focus == 'content_adjustment':
            instructions.append("Focus on content modifications: add, remove, or restructure information as requested")
        elif primary_focus == 'tone_adjustment':
            instructions.append("Adjust tone and style to match user preferences while maintaining professionalism")
        elif primary_focus == 'structure_adjustment':
            instructions.append("Reorganize content structure, flow, and presentation for better clarity")
        elif primary_focus == 'clarity_improvement':
            instructions.append("Enhance clarity and explanation, remove ambiguity, add specific details")
        elif primary_focus == 'framework_alignment':
            instructions.append("Strengthen KDKA and PRRR framework integration and alignment")
        else:
            instructions.append("Make general improvements based on user feedback")
        
        # Add specific pattern instructions
        for pattern_type, keywords in strategy['detected_patterns'].items():
            if pattern_type == 'framework_alignment':
                instructions.append(f"Pay special attention to {', '.join(keywords)} framework elements")
            elif pattern_type == 'clarity_improvement':
                instructions.append("Ensure all explanations are clear and unambiguous")
        
        # Add urgency-based instructions
        if strategy['urgency'] == 'high':
            instructions.append("PRIORITY: Address these concerns as primary objectives")
        elif strategy['urgency'] == 'medium':
            instructions.append("Important: Give significant attention to these improvements")
        
        # Add major revision flag
        if strategy['requires_major_revision']:
            instructions.append("NOTE: This requires substantial revision - don't hesitate to significantly restructure if needed")
        
        return "\\n".join(f"• {instruction}" for instruction in instructions)
    
    def _get_refinement_context(self) -> str:
        """Get relevant context for refinement"""
        context = self.orchestrator.conversation_state
        
        # Get recent conversation history
        recent_history = context.conversation_history[-5:] if context.conversation_history else []
        
        # Get approved outputs from other agents
        approved_outputs = {}
        for agent_id, outputs in context.agent_outputs.items():
            for output in outputs:
                if output.is_approved:
                    approved_outputs[agent_id] = {
                        'phase': output.phase,
                        'summary': output.content[:200] + '...' if len(output.content) > 200 else output.content
                    }
        
        context_info = f"""
Recent Conversation:
{[f"{turn['speaker']}: {turn['message'][:100]}..." for turn in recent_history]}

Approved Outputs from Other Agents:
{approved_outputs}

Current Phase: {context.current_phase}
Total Refinement Cycles This Phase: {sum(1 for k, v in context.refinement_cycles.items() if context.current_phase in k)}
"""
        
        return context_info
    
    async def _execute_refinement_task(self, agent_id: str, refined_task: str) -> str:
        """
        Execute the refinement task with the agent.
        
        Args:
            agent_id: Agent identifier
            refined_task: Refinement task description
            
        Returns:
            str: Refined output content
        """
        # This will use the same agent execution mechanism as the main orchestrator
        # For now, use the existing _execute_agent_task method
        
        agent = self.orchestrator.agents[agent_id]
        context = self.orchestrator._prepare_agent_context(agent_id)
        
        # Execute refined task
        refined_result = await self.orchestrator._execute_agent_task(
            agent, refined_task, context
        )
        
        return refined_result
    
    def get_refinement_history(
        self,
        agent_id: str,
        phase_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get refinement history for an agent in a phase.
        
        Args:
            agent_id: Agent identifier
            phase_id: Phase identifier
            
        Returns:
            List of refinement history entries
        """
        if not self.orchestrator.conversation_state:
            return []
        
        phase_id = phase_id or self.orchestrator.conversation_state.current_phase
        
        if agent_id not in self.orchestrator.conversation_state.agent_outputs:
            return []
        
        outputs = self.orchestrator.conversation_state.agent_outputs[agent_id]
        phase_outputs = [output for output in outputs if output.phase == phase_id]
        
        history = []
        for i, output in enumerate(phase_outputs):
            history.append({
                'version': output.version,
                'timestamp': output.timestamp.isoformat(),
                'user_feedback': output.user_feedback,
                'refinement_notes': output.refinement_notes,
                'is_approved': output.is_approved,
                'content_preview': output.content[:200] + '...' if len(output.content) > 200 else output.content
            })
        
        return history
    
    def analyze_refinement_convergence(
        self,
        agent_id: str,
        phase_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze if refinements are converging toward user satisfaction.
        
        Args:
            agent_id: Agent identifier
            phase_id: Phase identifier
            
        Returns:
            Dict with convergence analysis
        """
        history = self.get_refinement_history(agent_id, phase_id)
        
        if len(history) < 2:
            return {
                'convergence_status': 'insufficient_data',
                'recommendation': 'continue_refinement'
            }
        
        # Analyze feedback patterns over time
        feedback_lengths = [len(' '.join(entry['user_feedback'])) for entry in history]
        refinement_counts = [len(entry['refinement_notes']) for entry in history]
        
        # Simple convergence heuristics
        is_converging = (
            len(history) > 1 and
            feedback_lengths[-1] < feedback_lengths[0] and  # Feedback getting shorter
            refinement_counts[-1] > refinement_counts[0]     # More refinement cycles
        )
        
        analysis = {
            'convergence_status': 'converging' if is_converging else 'diverging',
            'total_cycles': len(history),
            'feedback_trend': 'decreasing' if feedback_lengths[-1] < feedback_lengths[0] else 'increasing',
            'recommendation': 'continue_refinement' if is_converging else 'consider_fresh_start'
        }
        
        if len(history) >= self.max_refinement_cycles:
            analysis['recommendation'] = 'force_approval_or_restart'
        
        return analysis