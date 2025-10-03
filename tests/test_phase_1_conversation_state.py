"""
Test Suite: Phase 1.1 - Conversation State Manager

Tests for ConversationState, AgentOutput, and PhaseState classes.
Validates state tracking, serialization, and frontend integration readiness.
"""

import unittest
import json
from datetime import datetime, timedelta
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.conversation_state import (
    ConversationState, 
    AgentOutput, 
    PhaseState, 
    PhaseStatus, 
    AgentStatus
)


class TestConversationState(unittest.TestCase):
    """Test conversation state management functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_course_request = {
            'course_title': 'Test Course',
            'course_description': 'A test course for validation',
            'course_credits': 3,
            'course_duration_weeks': 16,
            'course_level': 'Undergraduate'
        }
        
        self.conversation_state = ConversationState(
            session_id="test_session_123",
            course_request=self.sample_course_request
        )
    
    def test_conversation_state_initialization(self):
        """Test proper initialization of conversation state"""
        self.assertEqual(self.conversation_state.session_id, "test_session_123")
        self.assertEqual(self.conversation_state.course_request, self.sample_course_request)
        self.assertIsNone(self.conversation_state.current_phase)
        self.assertIsNone(self.conversation_state.active_agent)
        self.assertEqual(len(self.conversation_state.phases), 7)  # Expected number of phases
        
        # Verify phase initialization
        expected_phases = [
            'course_overview', 'foundation_design', 'content_creation',
            'technical_design', 'quality_review', 'ethical_audit', 'final_integration'
        ]
        for phase_id in expected_phases:
            self.assertIn(phase_id, self.conversation_state.phases)
            self.assertEqual(self.conversation_state.phases[phase_id].status, PhaseStatus.PENDING)
    
    def test_conversation_turn_management(self):
        """Test conversation turn tracking"""
        # Add conversation turns
        self.conversation_state.add_conversation_turn("user", "Hello, I want to create a course")
        self.conversation_state.add_conversation_turn("coordinator", "Great! Let's start the design process")
        
        # Verify turns were added
        self.assertEqual(len(self.conversation_state.conversation_history), 2)
        
        # Verify turn structure
        first_turn = self.conversation_state.conversation_history[0]
        self.assertEqual(first_turn['speaker'], "user")
        self.assertEqual(first_turn['message'], "Hello, I want to create a course")
        self.assertIn('timestamp', first_turn)
        
        # Verify last activity update
        self.assertIsNotNone(self.conversation_state.last_activity)
    
    def test_phase_management(self):
        """Test phase state transitions"""
        # Set current phase
        self.conversation_state.set_current_phase('foundation_design')
        
        # Verify phase activation
        self.assertEqual(self.conversation_state.current_phase, 'foundation_design')
        foundation_phase = self.conversation_state.phases['foundation_design']
        self.assertEqual(foundation_phase.status, PhaseStatus.ACTIVE)
        self.assertIsNotNone(foundation_phase.started_at)
        
        # Test phase transition
        self.conversation_state.set_current_phase('content_creation')
        
        # Verify previous phase completed
        self.assertEqual(foundation_phase.status, PhaseStatus.COMPLETED)
        self.assertIsNotNone(foundation_phase.completed_at)
        
        # Verify new phase activated
        content_phase = self.conversation_state.phases['content_creation']
        self.assertEqual(content_phase.status, PhaseStatus.ACTIVE)
        self.assertEqual(self.conversation_state.current_phase, 'content_creation')
    
    def test_agent_state_management(self):
        """Test agent state tracking"""
        # Set active agent
        self.conversation_state.set_active_agent('ipdai', AgentStatus.WORKING)
        
        # Verify agent state
        self.assertEqual(self.conversation_state.active_agent, 'ipdai')
        self.assertEqual(self.conversation_state.agents['ipdai']['status'], AgentStatus.WORKING.value)
        self.assertIn('last_activity', self.conversation_state.agents['ipdai'])
        
        # Update agent status
        self.conversation_state.set_active_agent('ipdai', AgentStatus.COMPLETED)
        self.assertEqual(self.conversation_state.agents['ipdai']['status'], AgentStatus.COMPLETED.value)
    
    def test_agent_output_management(self):
        """Test agent output tracking and versioning"""
        # Create sample agent output
        output1 = AgentOutput(
            agent_id='ipdai',
            agent_name='Instructional Planning Agent',
            phase='foundation_design',
            content='Initial course foundation design...',
            timestamp=datetime.now()
        )
        
        # Add output
        self.conversation_state.add_agent_output(output1)
        
        # Verify output added
        self.assertIn('ipdai', self.conversation_state.agent_outputs)
        self.assertEqual(len(self.conversation_state.agent_outputs['ipdai']), 1)
        self.assertEqual(output1.version, 1)
        
        # Add second version
        output2 = AgentOutput(
            agent_id='ipdai',
            agent_name='Instructional Planning Agent',
            phase='foundation_design',
            content='Refined course foundation design...',
            timestamp=datetime.now()
        )
        
        self.conversation_state.add_agent_output(output2)
        
        # Verify versioning
        self.assertEqual(len(self.conversation_state.agent_outputs['ipdai']), 2)
        self.assertEqual(output2.version, 2)
        
        # Test latest output retrieval
        latest = self.conversation_state.get_latest_output('ipdai', 'foundation_design')
        self.assertEqual(latest, output2)
        self.assertEqual(latest.version, 2)
    
    def test_user_feedback_and_approvals(self):
        """Test user feedback and approval tracking"""
        # Create and add agent output
        output = AgentOutput(
            agent_id='ipdai',
            agent_name='Instructional Planning Agent',
            phase='foundation_design',
            content='Course foundation content...',
            timestamp=datetime.now()
        )
        self.conversation_state.add_agent_output(output)
        
        # Add user feedback
        self.conversation_state.add_user_feedback('ipdai', 'Please add more detail to learning objectives')
        
        # Verify feedback added
        updated_output = self.conversation_state.get_latest_output('ipdai', 'foundation_design')
        self.assertEqual(len(updated_output.user_feedback), 1)
        self.assertEqual(updated_output.user_feedback[0], 'Please add more detail to learning objectives')
        
        # Test approval
        self.assertFalse(updated_output.is_approved)
        self.conversation_state.approve_output('ipdai', 'foundation_design')
        
        # Verify approval
        approved_output = self.conversation_state.get_latest_output('ipdai', 'foundation_design')
        self.assertTrue(approved_output.is_approved)
        self.assertIn('ipdai_foundation_design', self.conversation_state.user_approvals)
    
    def test_context_memory_management(self):
        """Test context memory functionality"""
        # Test global context
        self.conversation_state.update_context_memory('course_focus', 'AI and Machine Learning', 'global')
        global_value = self.conversation_state.get_context_memory('course_focus', 'global')
        self.assertEqual(global_value, 'AI and Machine Learning')
        
        # Test agent context
        self.conversation_state.set_active_agent('ipdai')
        self.conversation_state.update_context_memory('specialization', 'learning_objectives', 'agent')
        agent_value = self.conversation_state.get_context_memory('specialization', 'agent')
        self.assertEqual(agent_value, 'learning_objectives')
        
        # Test phase context
        self.conversation_state.set_current_phase('foundation_design')
        self.conversation_state.update_context_memory('framework_focus', 'KDKA', 'phase')
        phase_value = self.conversation_state.get_context_memory('framework_focus', 'phase')
        self.assertEqual(phase_value, 'KDKA')
    
    def test_progress_tracking(self):
        """Test phase progress calculation"""
        # Initially no phases completed
        progress = self.conversation_state.get_phase_progress()
        self.assertEqual(progress['completed_phases'], 0)
        self.assertEqual(progress['progress_percentage'], 0)
        
        # Complete some phases
        self.conversation_state.phases['course_overview'].status = PhaseStatus.COMPLETED
        self.conversation_state.phases['foundation_design'].status = PhaseStatus.COMPLETED
        
        progress = self.conversation_state.get_phase_progress()
        self.assertEqual(progress['completed_phases'], 2)
        expected_percentage = int((2 / 7) * 100)  # 2 out of 7 phases
        self.assertEqual(progress['progress_percentage'], expected_percentage)
    
    def test_serialization_for_frontend(self):
        """Test state serialization for frontend integration"""
        # Add some data
        self.conversation_state.add_conversation_turn("user", "Test message")
        self.conversation_state.set_current_phase('foundation_design')
        self.conversation_state.set_active_agent('ipdai')
        
        # Serialize to dict
        state_dict = self.conversation_state.to_dict()
        
        # Verify essential fields
        self.assertIn('session_id', state_dict)
        self.assertIn('conversation_history', state_dict)
        self.assertIn('current_phase', state_dict)
        self.assertIn('active_agent', state_dict)
        self.assertIn('phases', state_dict)
        self.assertIn('progress', state_dict)
        
        # Verify JSON serializable
        json_str = json.dumps(state_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        reconstructed_data = json.loads(json_str)
        self.assertEqual(reconstructed_data['session_id'], self.conversation_state.session_id)
    
    def test_refinement_cycle_tracking(self):
        """Test refinement cycle counting"""
        # First create an agent output
        output = AgentOutput(
            agent_id='ipdai',
            agent_name='Instructional Planning Agent',
            phase='foundation_design',
            content='Initial course foundation design...',
            timestamp=datetime.now()
        )
        self.conversation_state.add_agent_output(output)
        
        # Add initial feedback
        self.conversation_state.add_user_feedback('ipdai', 'First feedback', 'foundation_design')
        
        # Verify refinement cycle incremented
        cycle_key = 'ipdai_foundation_design'
        self.assertEqual(self.conversation_state.refinement_cycles[cycle_key], 1)
        
        # Add more feedback
        self.conversation_state.add_user_feedback('ipdai', 'Second feedback', 'foundation_design')
        self.assertEqual(self.conversation_state.refinement_cycles[cycle_key], 2)
    
    def test_pending_user_input_management(self):
        """Test pending user input state"""
        # Set pending input
        input_data = {
            'type': 'approval',
            'message': 'Please approve the course foundation',
            'options': ['approve', 'request_changes']
        }
        
        self.conversation_state.set_pending_user_input(input_data)
        self.assertEqual(self.conversation_state.pending_user_input, input_data)
        
        # Clear pending input
        self.conversation_state.clear_pending_user_input()
        self.assertIsNone(self.conversation_state.pending_user_input)


class TestAgentOutput(unittest.TestCase):
    """Test AgentOutput functionality"""
    
    def test_agent_output_creation(self):
        """Test agent output creation and serialization"""
        output = AgentOutput(
            agent_id='test_agent',
            agent_name='Test Agent',
            phase='test_phase',
            content='Test content for agent output',
            timestamp=datetime.now(),
            metadata={'test_key': 'test_value'}
        )
        
        # Test properties
        self.assertEqual(output.agent_id, 'test_agent')
        self.assertEqual(output.version, 1)
        self.assertFalse(output.is_approved)
        self.assertEqual(len(output.user_feedback), 0)
        
        # Test serialization
        output_dict = output.to_dict()
        self.assertIn('agent_id', output_dict)
        self.assertIn('timestamp', output_dict)
        self.assertIn('metadata', output_dict)
        
        # Test deserialization
        reconstructed = AgentOutput.from_dict(output_dict)
        self.assertEqual(reconstructed.agent_id, output.agent_id)
        self.assertEqual(reconstructed.content, output.content)


class TestPhaseState(unittest.TestCase):
    """Test PhaseState functionality"""
    
    def test_phase_state_creation(self):
        """Test phase state creation and management"""
        phase = PhaseState(
            phase_id='test_phase',
            phase_name='Test Phase',
            status=PhaseStatus.PENDING,
            assigned_agents=['agent1', 'agent2']
        )
        
        # Test initial state
        self.assertEqual(phase.phase_id, 'test_phase')
        self.assertEqual(phase.status, PhaseStatus.PENDING)
        self.assertEqual(len(phase.assigned_agents), 2)
        self.assertEqual(len(phase.active_agents), 0)
        self.assertIsNone(phase.started_at)
        
        # Test serialization
        phase_dict = phase.to_dict()
        self.assertIn('phase_id', phase_dict)
        self.assertIn('status', phase_dict)
        self.assertEqual(phase_dict['status'], PhaseStatus.PENDING.value)


def run_phase_1_tests():
    """Run all Phase 1 tests and return results"""
    print("🧪 Running Phase 1.1 Tests: Conversation State Manager")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestConversationState, TestAgentOutput, TestPhaseState]
    
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
            print(f"   - {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    run_phase_1_tests()