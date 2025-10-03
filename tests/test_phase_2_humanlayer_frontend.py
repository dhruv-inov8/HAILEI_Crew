"""
Test Suite: Phase 1.4 - HumanLayer Frontend Integration

Tests for FrontendHumanLayer and ConversationManager classes.
Validates frontend-ready human interaction and WebSocket compatibility.
"""

import unittest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_interface.humanlayer_frontend import (
    FrontendHumanLayer,
    InteractionRequest,
    InteractionResponse,
    InteractionType
)
from human_interface.conversation_manager import (
    ConversationManager,
    ConversationTurn,
    ConversationContext
)


class TestFrontendHumanLayer(unittest.TestCase):
    """Test FrontendHumanLayer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock HumanLayer to avoid external dependencies
        with patch('human_interface.humanlayer_frontend.HumanLayer'):
            self.frontend_humanlayer = FrontendHumanLayer(
                enable_websocket=False,  # Disable WebSocket for testing
                enable_logging=False
            )
        
        self.session_id = "test_session_123"
    
    def test_interaction_request_creation(self):
        """Test interaction request creation and serialization"""
        request = InteractionRequest(
            request_id="test_request_1",
            session_id=self.session_id,
            interaction_type=InteractionType.APPROVAL,
            title="Test Approval",
            message="Please approve this test",
            options=["approve", "deny"],
            timeout_seconds=300
        )
        
        # Test properties
        self.assertEqual(request.request_id, "test_request_1")
        self.assertEqual(request.interaction_type, InteractionType.APPROVAL)
        self.assertEqual(len(request.options), 2)
        
        # Test serialization
        request_dict = request.to_dict()
        self.assertIn('request_id', request_dict)
        self.assertIn('interaction_type', request_dict)
        self.assertEqual(request_dict['interaction_type'], 'approval')
        
        # Test JSON serialization
        json_str = json.dumps(request_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        reconstructed = InteractionRequest.from_dict(request_dict)
        self.assertEqual(reconstructed.request_id, request.request_id)
        self.assertEqual(reconstructed.interaction_type, request.interaction_type)
    
    def test_interaction_response_creation(self):
        """Test interaction response creation"""
        response = InteractionResponse(
            request_id="test_request_1",
            session_id=self.session_id,
            response_type="approval",
            response_data=True,
            user_id="test_user"
        )
        
        # Test properties
        self.assertEqual(response.request_id, "test_request_1")
        self.assertEqual(response.response_data, True)
        self.assertEqual(response.user_id, "test_user")
        
        # Test serialization
        response_dict = response.to_dict()
        self.assertIn('request_id', response_dict)
        self.assertIn('response_data', response_dict)
        self.assertIn('timestamp', response_dict)
    
    def test_event_callback_registration(self):
        """Test event callback system"""
        callback_called = False
        test_data = {}
        
        def test_callback(data):
            nonlocal callback_called, test_data
            callback_called = True
            test_data = data
        
        # Register callback
        self.frontend_humanlayer.register_event_callback('test_event', test_callback)
        
        # Emit event
        event_data = {'test_key': 'test_value'}
        self.frontend_humanlayer.emit_event('test_event', event_data)
        
        # Verify callback was called
        self.assertTrue(callback_called)
        self.assertEqual(test_data, event_data)
    
    def test_response_validation(self):
        """Test response validation logic"""
        # Create test request
        approval_request = InteractionRequest(
            request_id="test_approval",
            session_id=self.session_id,
            interaction_type=InteractionType.APPROVAL,
            title="Test",
            message="Test approval",
            options=["approve", "deny"]
        )
        
        # Test valid approval responses
        self.assertTrue(self.frontend_humanlayer._validate_response(approval_request, True))
        self.assertTrue(self.frontend_humanlayer._validate_response(approval_request, "approve"))
        
        # Test choice request
        choice_request = InteractionRequest(
            request_id="test_choice",
            session_id=self.session_id,
            interaction_type=InteractionType.CHOICE,
            title="Test",
            message="Test choice",
            options=["option1", "option2", "option3"]
        )
        
        # Test valid choice responses
        self.assertTrue(self.frontend_humanlayer._validate_response(choice_request, "option1"))
        self.assertTrue(self.frontend_humanlayer._validate_response(choice_request, "option2"))
        
        # Test feedback request
        feedback_request = InteractionRequest(
            request_id="test_feedback",
            session_id=self.session_id,
            interaction_type=InteractionType.FEEDBACK,
            title="Test",
            message="Test feedback"
        )
        
        # Test valid feedback responses
        self.assertTrue(self.frontend_humanlayer._validate_response(feedback_request, "This is good feedback"))
        self.assertTrue(self.frontend_humanlayer._validate_response(feedback_request, ""))  # Allow empty feedback
    
    def test_pending_interaction_management(self):
        """Test pending interaction tracking"""
        # Initially no pending interactions
        pending = self.frontend_humanlayer.get_pending_interactions(self.session_id)
        self.assertEqual(len(pending), 0)
        
        # Create and store pending interaction
        request = InteractionRequest(
            request_id="test_pending",
            session_id=self.session_id,
            interaction_type=InteractionType.APPROVAL,
            title="Test",
            message="Test pending interaction"
        )
        
        self.frontend_humanlayer.pending_interactions[request.request_id] = request
        
        # Verify pending interaction tracked
        pending = self.frontend_humanlayer.get_pending_interactions(self.session_id)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['request_id'], "test_pending")
    
    def test_response_submission(self):
        """Test response submission workflow"""
        # Create pending request
        request = InteractionRequest(
            request_id="test_submit",
            session_id=self.session_id,
            interaction_type=InteractionType.APPROVAL,
            title="Test",
            message="Test submission"
        )
        
        self.frontend_humanlayer.pending_interactions[request.request_id] = request
        
        # Submit response
        success = self.frontend_humanlayer.submit_response(
            request_id="test_submit",
            response_data=True,
            user_id="test_user"
        )
        
        # Verify submission success
        self.assertTrue(success)
        
        # Verify response stored
        self.assertIn("test_submit", self.frontend_humanlayer.interaction_responses)
        
        response = self.frontend_humanlayer.interaction_responses["test_submit"]
        self.assertEqual(response.response_data, True)
        self.assertEqual(response.user_id, "test_user")
    
    def test_session_state_tracking(self):
        """Test session state management"""
        # Get initial state
        state = self.frontend_humanlayer.get_session_state(self.session_id)
        
        # Verify state structure
        self.assertIn('session_id', state)
        self.assertIn('pending_interactions', state)
        self.assertIn('websocket_connected', state)
        self.assertIn('timestamp', state)
        
        # Verify session ID
        self.assertEqual(state['session_id'], self.session_id)
        self.assertFalse(state['websocket_connected'])  # WebSocket disabled in test


class TestConversationManager(unittest.TestCase):
    """Test ConversationManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock FrontendHumanLayer
        with patch('human_interface.conversation_manager.FrontendHumanLayer'):
            mock_frontend = Mock()
            self.conversation_manager = ConversationManager(
                frontend_humanlayer=mock_frontend,
                enable_logging=False
            )
        
        self.session_id = "test_session_456"
    
    def test_conversation_turn_creation(self):
        """Test conversation turn creation"""
        turn = ConversationTurn(
            turn_id="turn_1",
            session_id=self.session_id,
            speaker="user",
            message="Hello, I want to create a course",
            timestamp=datetime.now(),
            context={'intent': 'course_creation'}
        )
        
        # Test properties
        self.assertEqual(turn.speaker, "user")
        self.assertEqual(turn.session_id, self.session_id)
        self.assertIn('intent', turn.context)
        
        # Test serialization
        turn_dict = turn.to_dict()
        self.assertIn('turn_id', turn_dict)
        self.assertIn('speaker', turn_dict)
        self.assertIn('timestamp', turn_dict)
        
        # Verify JSON serializable
        json_str = json.dumps(turn_dict)
        self.assertIsInstance(json_str, str)
    
    def test_conversation_context_creation(self):
        """Test conversation context management"""
        # Create sample turns
        turns = [
            ConversationTurn(
                turn_id="turn_1",
                session_id=self.session_id,
                speaker="user",
                message="I want to create an AI course",
                timestamp=datetime.now()
            ),
            ConversationTurn(
                turn_id="turn_2", 
                session_id=self.session_id,
                speaker="coordinator",
                message="Great! Let's start with the course foundation",
                timestamp=datetime.now()
            )
        ]
        
        context = ConversationContext(
            session_id=self.session_id,
            current_speaker="coordinator",
            conversation_summary="User wants to create AI course",
            recent_turns=turns,
            active_topics=["AI", "course_creation"],
            user_preferences={"communication_style": "detailed"}
        )
        
        # Test properties
        self.assertEqual(context.session_id, self.session_id)
        self.assertEqual(len(context.recent_turns), 2)
        self.assertEqual(len(context.active_topics), 2)
        
        # Test serialization
        context_dict = context.to_dict()
        self.assertIn('session_id', context_dict)
        self.assertIn('recent_turns', context_dict)
        self.assertIn('active_topics', context_dict)
    
    def test_conversation_start(self):
        """Test conversation initialization"""
        # Start conversation
        greeting = self.conversation_manager.start_conversation(
            self.session_id,
            initial_context={'course_type': 'AI'}
        )
        
        # Verify conversation started
        self.assertIn(self.session_id, self.conversation_manager.conversations)
        self.assertIn(self.session_id, self.conversation_manager.conversation_contexts)
        
        # Verify greeting message
        self.assertIsInstance(greeting, str)
        self.assertIn("HAILEI", greeting)
    
    def test_conversation_turn_addition(self):
        """Test adding conversation turns"""
        # Start conversation
        self.conversation_manager.start_conversation(self.session_id)
        
        # Add turns
        turn1 = self.conversation_manager.add_turn(
            self.session_id,
            "user",
            "I want to create a machine learning course",
            context={'intent': 'course_creation'}
        )
        
        turn2 = self.conversation_manager.add_turn(
            self.session_id,
            "coordinator", 
            "Excellent! Let's design your ML course",
            context={'response_to': turn1.turn_id}
        )
        
        # Verify turns added
        conversation = self.conversation_manager.conversations[self.session_id]
        self.assertEqual(len(conversation), 2)
        
        # Verify turn properties
        self.assertEqual(conversation[0].speaker, "user")
        self.assertEqual(conversation[1].speaker, "coordinator")
        
        # Verify context updates
        context = self.conversation_manager.conversation_contexts[self.session_id]
        self.assertEqual(context.current_speaker, "coordinator")
        self.assertEqual(len(context.recent_turns), 2)
    
    def test_agent_context_generation(self):
        """Test agent context generation"""
        # Start conversation and add turns
        self.conversation_manager.start_conversation(self.session_id)
        self.conversation_manager.add_turn(self.session_id, "user", "Create AI course")
        self.conversation_manager.add_turn(self.session_id, "coordinator", "Starting AI course design")
        
        # Update agent memory
        self.conversation_manager.update_agent_memory(
            self.session_id,
            "ipdai",
            {"specialization": "learning_objectives", "framework": "KDKA"}
        )
        
        # Get agent context
        agent_context = self.conversation_manager.get_agent_context(self.session_id, "ipdai")
        
        # Verify context structure
        self.assertIn('conversation_summary', agent_context)
        self.assertIn('recent_conversation', agent_context)
        self.assertIn('active_topics', agent_context)
        self.assertIn('agent_memory', agent_context)
        self.assertIn('conversation_metadata', agent_context)
        
        # Verify agent memory
        self.assertEqual(agent_context['agent_memory']['specialization'], "learning_objectives")
        self.assertEqual(agent_context['agent_memory']['framework'], "KDKA")
    
    def test_user_preference_learning(self):
        """Test user preference detection"""
        # Start conversation
        self.conversation_manager.start_conversation(self.session_id)
        
        # Add detailed user message
        detailed_message = "I want to create a comprehensive machine learning course that covers neural networks, deep learning, computer vision, and natural language processing with practical hands-on projects."
        
        self.conversation_manager.add_turn(self.session_id, "user", detailed_message)
        
        # Check learned preferences
        preferences = self.conversation_manager.user_preferences[self.session_id]
        self.assertEqual(preferences.get('communication_style'), 'detailed')
        
        # Add brief user message
        brief_message = "Yes"
        self.conversation_manager.add_turn(self.session_id, "user", brief_message)
        
        # Verify preference updated
        preferences = self.conversation_manager.user_preferences[self.session_id]
        self.assertEqual(preferences.get('communication_style'), 'concise')
    
    def test_conversation_history_retrieval(self):
        """Test conversation history functionality"""
        # Start conversation and add multiple turns
        self.conversation_manager.start_conversation(self.session_id)
        
        for i in range(5):
            self.conversation_manager.add_turn(
                self.session_id,
                "user" if i % 2 == 0 else "coordinator",
                f"Message {i + 1}"
            )
        
        # Get full history
        full_history = self.conversation_manager.get_conversation_history(self.session_id)
        self.assertEqual(len(full_history), 5)
        
        # Get limited history
        limited_history = self.conversation_manager.get_conversation_history(self.session_id, limit=3)
        self.assertEqual(len(limited_history), 3)
        
        # Get user messages only
        user_history = self.conversation_manager.get_conversation_history(
            self.session_id, 
            speaker_filter="user"
        )
        self.assertEqual(len(user_history), 3)  # Messages 1, 3, 5
    
    def test_conversation_statistics(self):
        """Test conversation statistics generation"""
        # Start conversation and add turns
        self.conversation_manager.start_conversation(self.session_id)
        
        self.conversation_manager.add_turn(self.session_id, "user", "Hello")
        self.conversation_manager.add_turn(self.session_id, "coordinator", "Hi there!")
        self.conversation_manager.add_turn(self.session_id, "user", "I want to create a course")
        
        # Get statistics
        stats = self.conversation_manager.get_conversation_stats(self.session_id)
        
        # Verify statistics
        self.assertEqual(stats['total_turns'], 3)
        self.assertEqual(stats['user_turns'], 2)
        self.assertEqual(stats['agent_turns'], 1)
        self.assertIn('start_time', stats)
        self.assertIn('last_activity', stats)


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()


class TestAsyncFunctionality(AsyncTestCase):
    """Test async functionality of FrontendHumanLayer"""
    
    def setUp(self):
        super().setUp()
        with patch('human_interface.humanlayer_frontend.HumanLayer'):
            self.frontend_humanlayer = FrontendHumanLayer(
                enable_websocket=False,
                enable_logging=False
            )
        self.session_id = "test_async_session"
    
    def test_timeout_handling(self):
        """Test timeout handling in interactions"""
        async def test_timeout():
            # Create request with short timeout
            request = InteractionRequest(
                request_id="timeout_test",
                session_id=self.session_id,
                interaction_type=InteractionType.APPROVAL,
                title="Timeout Test",
                message="This will timeout",
                timeout_seconds=0.1  # Very short timeout
            )
            
            # Test timeout behavior
            try:
                await self.frontend_humanlayer._wait_for_response(request)
                self.fail("Should have timed out")
            except asyncio.TimeoutError:
                pass  # Expected
        
        self.loop.run_until_complete(test_timeout())
    
    def test_default_response_generation(self):
        """Test default response generation for timeouts"""
        # Test approval default
        approval_request = InteractionRequest(
            request_id="approval_timeout",
            session_id=self.session_id,
            interaction_type=InteractionType.APPROVAL,
            title="Test",
            message="Test"
        )
        
        default_response = self.frontend_humanlayer._get_default_response(approval_request)
        self.assertEqual(default_response.response_data, False)
        self.assertEqual(default_response.response_type, "timeout")
        
        # Test choice default
        choice_request = InteractionRequest(
            request_id="choice_timeout",
            session_id=self.session_id,
            interaction_type=InteractionType.CHOICE,
            title="Test",
            message="Test",
            options=["option1", "option2"]
        )
        
        default_response = self.frontend_humanlayer._get_default_response(choice_request)
        self.assertEqual(default_response.response_data, "option1")  # First option


def run_phase_2_tests():
    """Run all Phase 2 tests and return results"""
    print("🧪 Running Phase 1.4 Tests: HumanLayer Frontend Integration")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFrontendHumanLayer,
        TestConversationManager,
        TestAsyncFunctionality
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    run_phase_2_tests()