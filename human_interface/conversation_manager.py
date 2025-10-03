"""
HAILEI Conversation Manager

Manages multi-turn conversations and context preservation for frontend deployment.
Handles conversation flow, context injection, and state synchronization.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from .humanlayer_frontend import FrontendHumanLayer, InteractionType


@dataclass
class ConversationTurn:
    """Individual conversation turn with context"""
    turn_id: str
    session_id: str
    speaker: str  # 'user', 'agent', 'coordinator'
    message: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'turn_id': self.turn_id,
            'session_id': self.session_id,
            'speaker': self.speaker,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class ConversationContext:
    """Conversation context for agent awareness"""
    session_id: str
    current_speaker: str
    conversation_summary: str
    recent_turns: List[ConversationTurn]
    active_topics: List[str]
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    agent_memory: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for agent context"""
        return {
            'session_id': self.session_id,
            'current_speaker': self.current_speaker,
            'conversation_summary': self.conversation_summary,
            'recent_turns': [turn.to_dict() for turn in self.recent_turns],
            'active_topics': self.active_topics,
            'user_preferences': self.user_preferences,
            'agent_memory': self.agent_memory
        }


class ConversationManager:
    """
    Manages conversational flow and context for HAILEI agents.
    
    Features:
    - Multi-turn conversation tracking
    - Context preservation across agent switches
    - Conversation summarization for long sessions
    - User preference learning and adaptation
    - Agent memory management
    - Frontend conversation display formatting
    """
    
    def __init__(
        self,
        frontend_humanlayer: FrontendHumanLayer,
        max_context_turns: int = 20,
        enable_summarization: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize conversation manager.
        
        Args:
            frontend_humanlayer: Frontend HumanLayer instance
            max_context_turns: Maximum turns to keep in active context
            enable_summarization: Enable conversation summarization
            enable_logging: Enable detailed logging
        """
        self.frontend_humanlayer = frontend_humanlayer
        self.max_context_turns = max_context_turns
        self.enable_summarization = enable_summarization
        
        # Conversation storage
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.conversation_summaries: Dict[str, str] = {}
        
        # User preference tracking
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Agent memory per session
        self.agent_memories: Dict[str, Dict[str, Any]] = {}  # session_id -> agent_memory
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger('hailei_conversation_manager')
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
    
    def start_conversation(self, session_id: str, initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start new conversation session.
        
        Args:
            session_id: Session identifier
            initial_context: Initial context for conversation
            
        Returns:
            str: Conversation greeting message
        """
        # Initialize conversation storage
        self.conversations[session_id] = []
        self.conversation_summaries[session_id] = ""
        self.user_preferences[session_id] = {}
        self.agent_memories[session_id] = {}
        
        # Create initial context
        self.conversation_contexts[session_id] = ConversationContext(
            session_id=session_id,
            current_speaker="coordinator",
            conversation_summary="",
            recent_turns=[],
            active_topics=[],
            user_preferences=initial_context or {},
            agent_memory={}
        )
        
        self.logger.info(f"Started conversation for session: {session_id}")
        
        # Return greeting message
        return self._generate_greeting_message(session_id)
    
    def add_turn(
        self,
        session_id: str,
        speaker: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add conversation turn and update context.
        
        Args:
            session_id: Session identifier
            speaker: Speaker identifier
            message: Message content
            context: Additional context
            metadata: Additional metadata
            
        Returns:
            ConversationTurn: Created conversation turn
        """
        if session_id not in self.conversations:
            self.start_conversation(session_id)
        
        # Create turn
        turn = ConversationTurn(
            turn_id=f"{session_id}_{len(self.conversations[session_id])}",
            session_id=session_id,
            speaker=speaker,
            message=message,
            timestamp=datetime.now(),
            context=context or {},
            metadata=metadata or {}
        )
        
        # Add to conversation
        self.conversations[session_id].append(turn)
        
        # Update conversation context
        self._update_conversation_context(session_id, turn)
        
        # Learn user preferences if user message
        if speaker == "user":
            self._learn_user_preferences(session_id, message, context)
        
        self.logger.info(f"Added turn for session {session_id}: {speaker}")
        
        return turn
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get current conversation context for session"""
        return self.conversation_contexts.get(session_id)
    
    def get_agent_context(self, session_id: str, agent_id: str) -> Dict[str, Any]:
        """
        Get context for specific agent including conversation history.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            
        Returns:
            Dict: Comprehensive context for agent
        """
        if session_id not in self.conversation_contexts:
            return {}
        
        context = self.conversation_contexts[session_id]
        
        # Get agent-specific memory
        agent_memory = self.agent_memories[session_id].get(agent_id, {})
        
        # Build comprehensive context
        agent_context = {
            'conversation_summary': context.conversation_summary,
            'recent_conversation': [
                {
                    'speaker': turn.speaker,
                    'message': turn.message,
                    'timestamp': turn.timestamp.isoformat()
                }
                for turn in context.recent_turns[-10:]  # Last 10 turns
            ],
            'active_topics': context.active_topics,
            'user_preferences': context.user_preferences,
            'agent_memory': agent_memory,
            'conversation_metadata': {
                'total_turns': len(self.conversations.get(session_id, [])),
                'conversation_start': self.conversations[session_id][0].timestamp.isoformat() if session_id in self.conversations and self.conversations[session_id] else None,
                'last_activity': context.recent_turns[-1].timestamp.isoformat() if context.recent_turns else None
            }
        }
        
        return agent_context
    
    def update_agent_memory(self, session_id: str, agent_id: str, memory_update: Dict[str, Any]):
        """Update agent-specific memory"""
        if session_id not in self.agent_memories:
            self.agent_memories[session_id] = {}
        
        if agent_id not in self.agent_memories[session_id]:
            self.agent_memories[session_id][agent_id] = {}
        
        self.agent_memories[session_id][agent_id].update(memory_update)
        
        # Also update conversation context
        if session_id in self.conversation_contexts:
            self.conversation_contexts[session_id].agent_memory.update({
                agent_id: self.agent_memories[session_id][agent_id]
            })
        
        self.logger.info(f"Updated agent memory for {agent_id} in session {session_id}")
    
    async def conduct_guided_conversation(
        self,
        session_id: str,
        agent_id: str,
        initial_prompt: str,
        conversation_goals: List[str],
        max_turns: int = 10
    ) -> List[ConversationTurn]:
        """
        Conduct guided conversation between user and agent.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            initial_prompt: Initial prompt from agent
            conversation_goals: Goals for the conversation
            max_turns: Maximum conversation turns
            
        Returns:
            List of conversation turns
        """
        conversation_turns = []
        
        # Add initial agent prompt
        agent_turn = self.add_turn(session_id, agent_id, initial_prompt, {
            'conversation_type': 'guided',
            'goals': conversation_goals
        })
        conversation_turns.append(agent_turn)
        
        # Conduct conversation loop
        for turn_number in range(max_turns):
            # Request user response
            user_response = await self.frontend_humanlayer.request_text_input(
                session_id=session_id,
                title=f"Conversation with {agent_id}",
                message=initial_prompt if turn_number == 0 else "Please continue the conversation:",
                context={
                    'agent_id': agent_id,
                    'turn_number': turn_number,
                    'conversation_goals': conversation_goals,
                    'recent_context': self.get_agent_context(session_id, agent_id)
                }
            )
            
            # Add user turn
            user_turn = self.add_turn(session_id, "user", user_response, {
                'conversation_type': 'guided',
                'turn_number': turn_number
            })
            conversation_turns.append(user_turn)
            
            # Check if conversation goals are met
            if self._check_conversation_goals(session_id, conversation_goals):
                break
            
            # Generate agent response (this would integrate with actual agent)
            agent_response = await self._generate_agent_response(
                session_id, agent_id, user_response, conversation_goals
            )
            
            # Add agent turn
            agent_turn = self.add_turn(session_id, agent_id, agent_response, {
                'conversation_type': 'guided',
                'turn_number': turn_number
            })
            conversation_turns.append(agent_turn)
        
        return conversation_turns
    
    async def handle_iterative_refinement(
        self,
        session_id: str,
        agent_id: str,
        content: str,
        refinement_prompt: str
    ) -> Tuple[str, bool]:  # (refined_content, is_approved)
        """
        Handle iterative refinement conversation.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            content: Content to refine
            refinement_prompt: Initial refinement prompt
            
        Returns:
            Tuple of (refined_content, is_approved)
        """
        # Present content for review
        review_response = await self.frontend_humanlayer.request_content_review(
            session_id=session_id,
            title=f"Review Output from {agent_id}",
            content=content,
            agent_name=agent_id,
            phase_name="refinement",
            context={
                'refinement_prompt': refinement_prompt,
                'original_content': content
            }
        )
        
        # Add review turn to conversation
        self.add_turn(session_id, "user", f"Review decision: {review_response}", {
            'interaction_type': 'content_review',
            'agent_id': agent_id
        })
        
        # Handle review decision
        decision = review_response.get('decision', 'approve')
        
        if decision == 'approve':
            return content, True
        
        elif decision == 'request_changes':
            # Get specific feedback
            feedback = review_response.get('feedback', '')
            if not feedback:
                feedback = await self.frontend_humanlayer.request_feedback(
                    session_id=session_id,
                    title="Provide Feedback",
                    message="Please provide specific feedback for improvements:",
                    current_content=content
                )
            
            # Add feedback turn
            self.add_turn(session_id, "user", f"Feedback: {feedback}", {
                'interaction_type': 'refinement_feedback',
                'agent_id': agent_id
            })
            
            # Return for refinement (actual refinement would be handled by orchestrator)
            return feedback, False
        
        else:  # reject
            return "Content rejected by user", False
    
    def _update_conversation_context(self, session_id: str, turn: ConversationTurn):
        """Update conversation context with new turn"""
        if session_id not in self.conversation_contexts:
            return
        
        context = self.conversation_contexts[session_id]
        
        # Update current speaker
        context.current_speaker = turn.speaker
        
        # Add to recent turns
        context.recent_turns.append(turn)
        
        # Maintain maximum context size
        if len(context.recent_turns) > self.max_context_turns:
            context.recent_turns = context.recent_turns[-self.max_context_turns:]
        
        # Update active topics
        self._extract_topics_from_turn(turn, context)
        
        # Update conversation summary if enabled
        if self.enable_summarization and len(context.recent_turns) >= 10:
            context.conversation_summary = self._summarize_conversation(session_id)
    
    def _extract_topics_from_turn(self, turn: ConversationTurn, context: ConversationContext):
        """Extract topics from conversation turn"""
        # Simple keyword extraction (could be enhanced with NLP)
        keywords = [
            'course', 'learning', 'objectives', 'assessment', 'content',
            'framework', 'kdka', 'prrr', 'bloom', 'taxonomy', 'design',
            'technical', 'implementation', 'quality', 'ethics', 'accessibility'
        ]
        
        message_lower = turn.message.lower()
        found_topics = [keyword for keyword in keywords if keyword in message_lower]
        
        # Add new topics to active topics
        for topic in found_topics:
            if topic not in context.active_topics:
                context.active_topics.append(topic)
        
        # Maintain reasonable topic list size
        if len(context.active_topics) > 10:
            context.active_topics = context.active_topics[-10:]
    
    def _learn_user_preferences(self, session_id: str, message: str, context: Optional[Dict[str, Any]]):
        """Learn user preferences from interactions"""
        if session_id not in self.user_preferences:
            self.user_preferences[session_id] = {}
        
        preferences = self.user_preferences[session_id]
        
        # Learn communication style preferences
        if len(message.split()) > 20:
            preferences['communication_style'] = 'detailed'
        elif len(message.split()) < 5:
            preferences['communication_style'] = 'concise'
        
        # Learn feedback style
        if any(word in message.lower() for word in ['specific', 'detailed', 'precise']):
            preferences['feedback_style'] = 'detailed'
        elif any(word in message.lower() for word in ['brief', 'short', 'quick']):
            preferences['feedback_style'] = 'brief'
        
        # Update conversation context
        if session_id in self.conversation_contexts:
            self.conversation_contexts[session_id].user_preferences.update(preferences)
    
    def _summarize_conversation(self, session_id: str) -> str:
        """Generate conversation summary"""
        if session_id not in self.conversations:
            return ""
        
        turns = self.conversations[session_id]
        if not turns:
            return ""
        
        # Simple summarization (could be enhanced with AI)
        recent_turns = turns[-20:] if len(turns) > 20 else turns
        
        summary_points = []
        current_phase = None
        
        for turn in recent_turns:
            if 'phase' in turn.context:
                phase = turn.context['phase']
                if phase != current_phase:
                    summary_points.append(f"Entered {phase} phase")
                    current_phase = phase
            
            if turn.speaker == "user" and any(word in turn.message.lower() for word in ['approve', 'good', 'excellent']):
                summary_points.append(f"User approved {turn.context.get('agent_id', 'content')}")
            
            if 'refinement' in turn.context:
                summary_points.append(f"Refinement cycle for {turn.context.get('agent_id', 'agent')}")
        
        return "; ".join(summary_points[-5:])  # Last 5 major events
    
    def _generate_greeting_message(self, session_id: str) -> str:
        """Generate personalized greeting message"""
        return f"""
🎓 Welcome to HAILEI Course Design!

I'm your educational AI coordinator, ready to help you create an exceptional course design. 
I'll be working with our team of specialized agents to bring your vision to life.

Let's start by reviewing your course requirements and then begin our collaborative design process!
"""
    
    async def _generate_agent_response(
        self,
        session_id: str,
        agent_id: str,
        user_message: str,
        conversation_goals: List[str]
    ) -> str:
        """Generate agent response (placeholder for actual agent integration)"""
        # This would integrate with the actual agent system
        return f"[{agent_id}] Thank you for your input: '{user_message}'. Let me continue working on our goals: {', '.join(conversation_goals)}"
    
    def _check_conversation_goals(self, session_id: str, goals: List[str]) -> bool:
        """Check if conversation goals have been met"""
        # Simple heuristic - could be enhanced with AI
        if session_id not in self.conversations:
            return False
        
        recent_turns = self.conversations[session_id][-5:]
        user_turns = [turn for turn in recent_turns if turn.speaker == "user"]
        
        # Check if user has expressed satisfaction
        satisfaction_indicators = ['good', 'great', 'perfect', 'done', 'finished', 'approved']
        
        for turn in user_turns:
            if any(indicator in turn.message.lower() for indicator in satisfaction_indicators):
                return True
        
        return False
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        speaker_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for frontend display"""
        if session_id not in self.conversations:
            return []
        
        turns = self.conversations[session_id]
        
        # Apply speaker filter
        if speaker_filter:
            turns = [turn for turn in turns if turn.speaker == speaker_filter]
        
        # Apply limit
        if limit:
            turns = turns[-limit:]
        
        return [turn.to_dict() for turn in turns]
    
    def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get conversation statistics for analytics"""
        if session_id not in self.conversations:
            return {}
        
        turns = self.conversations[session_id]
        
        stats = {
            'total_turns': len(turns),
            'user_turns': len([t for t in turns if t.speaker == "user"]),
            'agent_turns': len([t for t in turns if t.speaker != "user"]),
            'start_time': turns[0].timestamp.isoformat() if turns else None,
            'last_activity': turns[-1].timestamp.isoformat() if turns else None,
            'active_topics': self.conversation_contexts[session_id].active_topics if session_id in self.conversation_contexts else [],
            'user_preferences': self.user_preferences.get(session_id, {}),
            'conversation_summary': self.conversation_summaries.get(session_id, "")
        }
        
        return stats