"""
Test Suite: End-to-End Conversational Workflow

Tests complete conversation flow from session creation to agent execution
and user interaction. Validates the full HAILEI conversational system.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import HAILEIOrchestrator, ConversationState
from agents.agent_wrappers import AgentExecutionResult


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end conversational workflow"""
    
    def setUp(self):
        """Set up test environment"""
        # Framework definitions
        self.frameworks = {
            'kdka': {
                'name': 'Knowledge, Delivery, Context, Assessment',
                'summary': 'Comprehensive instructional design framework focusing on knowledge transfer'
            },
            'prrr': {
                'name': 'Personal, Relatable, Relative, Real-world',
                'summary': 'Engagement framework ensuring content resonates with learners'
            }
        }
        
        # Sample course request
        self.course_request = {
            'course_title': 'Introduction to Machine Learning',
            'course_description': 'Comprehensive introduction to ML concepts and applications',
            'course_level': 'Undergraduate',
            'course_duration_weeks': 16,
            'course_credits': 3,
            'target_audience': 'Computer Science students',
            'prerequisites': 'Programming fundamentals, basic statistics'
        }
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    @patch('agents.agent_wrappers.AgentExecutor.execute_agent_task')
    def test_complete_conversation_workflow(self, mock_execute):
        """Test complete workflow from session creation to agent execution"""
        
        # Mock agent execution results
        def mock_agent_execution(agent_id, task_description, **kwargs):
            """Mock different agent responses based on agent_id"""
            if agent_id == "ipdai_agent":
                return AgentExecutionResult(
                    agent_id=agent_id,
                    task_description=task_description,
                    output=f"IPDAi: Created comprehensive course foundation for {kwargs.get('context', {}).get('course_request', {}).get('course_title', 'Unknown Course')} using KDKA framework. Developed learning objectives, assessment strategies, and knowledge delivery pathways.",
                    execution_time=2.1,
                    success=True,
                    metadata={'framework_used': 'KDKA', 'phase': 'foundation_design'}
                )
            elif agent_id == "cauthai_agent":
                return AgentExecutionResult(
                    agent_id=agent_id,
                    task_description=task_description,
                    output=f"CAuthAi: Developed engaging content modules for machine learning course using PRRR framework. Created real-world examples, personal connections, and relatable scenarios for better learning engagement.",
                    execution_time=3.2,
                    success=True,
                    metadata={'framework_used': 'PRRR', 'content_modules': 5}
                )
            else:
                return AgentExecutionResult(
                    agent_id=agent_id,
                    task_description=task_description,
                    output=f"Agent {agent_id}: Successfully completed task - {task_description}",
                    execution_time=1.5,
                    success=True
                )
        
        mock_execute.side_effect = mock_agent_execution
        
        async def test_workflow():
            # Initialize orchestrator
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks=self.frameworks,
                enable_logging=False
            )
            
            # Step 1: Create conversation session
            session = orchestrator.create_session(self.course_request)
            self.assertIsNotNone(session)
            self.assertEqual(session.course_request['course_title'], 'Introduction to Machine Learning')
            
            # Step 2: Add initial conversation turns
            session.add_conversation_turn(
                "user", 
                "I want to create a comprehensive machine learning course for undergraduates"
            )
            session.add_conversation_turn(
                "coordinator", 
                "Excellent! Let's design your ML course using our HAILEI framework. I'll start by activating our instructional planning specialist."
            )
            
            # Step 3: Activate first agent (IPDAi for foundation design)
            session.set_current_phase('foundation_design')
            
            ipdai_output = await orchestrator.activate_agent(
                agent_id="ipdai_agent",
                task_description="Create comprehensive course foundation using KDKA framework for machine learning course",
                context={'phase': 'foundation_design', 'focus': 'learning_objectives'}
            )
            
            # Verify agent output
            self.assertIsNotNone(ipdai_output)
            self.assertEqual(ipdai_output.agent_id, "ipdai_agent")
            self.assertIn("IPDAi", ipdai_output.content)
            self.assertIn("KDKA framework", ipdai_output.content)
            self.assertIn("Introduction to Machine Learning", ipdai_output.content)
            
            # Step 4: Add agent output to conversation state
            session.add_agent_output(ipdai_output)
            
            # Step 5: Simulate user feedback
            session.add_conversation_turn(
                "user",
                "The foundation looks good, but I'd like more emphasis on practical applications"
            )
            
            # Step 6: Add user feedback to agent output
            session.add_user_feedback(
                "ipdai_agent", 
                "Please add more emphasis on practical applications and real-world examples",
                "foundation_design"
            )
            
            # Step 7: Transition to next phase and activate content agent
            session.set_current_phase('content_creation')
            
            cauthai_output = await orchestrator.activate_agent(
                agent_id="cauthai_agent",
                task_description="Create engaging content modules using PRRR framework with emphasis on practical applications",
                context={'phase': 'content_creation', 'previous_feedback': 'emphasize practical applications'}
            )
            
            # Verify second agent output
            self.assertIsNotNone(cauthai_output)
            self.assertEqual(cauthai_output.agent_id, "cauthai_agent")
            self.assertIn("CAuthAi", cauthai_output.content)
            self.assertIn("PRRR framework", cauthai_output.content)
            self.assertIn("real-world examples", cauthai_output.content)
            
            # Step 8: Add second agent output
            session.add_agent_output(cauthai_output)
            
            # Step 9: Approve both outputs
            session.approve_output("ipdai_agent", "foundation_design")
            session.approve_output("cauthai_agent", "content_creation")
            
            # Step 10: Verify conversation state
            self.assertEqual(len(session.conversation_history), 3)  # user, coordinator, user feedback
            self.assertEqual(len(session.agent_outputs), 2)  # IPDAi and CAuthAi outputs
            self.assertEqual(len(session.user_approvals), 2)  # Both approved
            
            # Step 11: Test conversation context preparation
            context = orchestrator._prepare_agent_context("next_agent", {"additional": "context"})
            
            # Verify context includes previous work
            self.assertIn('previous_outputs', context)
            self.assertEqual(len(context['previous_outputs']), 2)
            self.assertIn('ipdai_agent', context['previous_outputs'])
            self.assertIn('cauthai_agent', context['previous_outputs'])
            
            # Step 12: Test progress tracking
            progress = session.get_phase_progress()
            self.assertGreater(progress['completed_phases'], 0)
            self.assertGreater(progress['progress_percentage'], 0)
            
            # Step 13: Test serialization for frontend
            session_dict = session.to_dict()
            self.assertIn('session_id', session_dict)
            self.assertIn('conversation_history', session_dict)
            self.assertIn('agent_outputs', session_dict)
            self.assertIn('progress', session_dict)
            
            return session, orchestrator
        
        # Run the workflow test
        session, orchestrator = self.loop.run_until_complete(test_workflow())
        
        # Final verification
        self.assertIsNotNone(session)
        self.assertIsNotNone(orchestrator)
        self.assertIn(session.session_id, orchestrator.active_sessions)
    
    def test_conversation_flow_with_refinement(self):
        """Test conversation flow with iterative refinement"""
        
        async def test_refinement():
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks=self.frameworks,
                enable_logging=False
            )
            
            # Create session
            session = orchestrator.create_session(self.course_request)
            session.set_current_phase('foundation_design')
            
            # Create initial agent output
            from orchestrator.conversation_state import AgentOutput
            initial_output = AgentOutput(
                agent_id="ipdai_agent",
                agent_name="Instructional Planning Agent",
                phase="foundation_design",
                content="Initial course foundation with basic learning objectives",
                timestamp=datetime.now()
            )
            session.add_agent_output(initial_output)
            
            # Add multiple rounds of feedback
            session.add_user_feedback("ipdai_agent", "Add more practical examples", "foundation_design")
            session.add_user_feedback("ipdai_agent", "Include assessment rubrics", "foundation_design")
            session.add_user_feedback("ipdai_agent", "Align with industry standards", "foundation_design")
            
            # Verify refinement tracking
            refinement_key = "ipdai_agent_foundation_design"
            self.assertEqual(session.refinement_cycles[refinement_key], 3)
            
            # Verify feedback is stored
            latest_output = session.get_latest_output("ipdai_agent", "foundation_design")
            self.assertEqual(len(latest_output.user_feedback), 3)
            self.assertIn("practical examples", latest_output.user_feedback[0])
            self.assertIn("assessment rubrics", latest_output.user_feedback[1])
            self.assertIn("industry standards", latest_output.user_feedback[2])
        
        self.loop.run_until_complete(test_refinement())
    
    def test_error_handling_in_workflow(self):
        """Test error handling throughout the workflow"""
        
        async def test_errors():
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks=self.frameworks,
                enable_logging=False
            )
            
            # Test session creation
            session = orchestrator.create_session(self.course_request)
            
            # Test invalid agent activation
            try:
                await orchestrator.activate_agent(
                    agent_id="nonexistent_agent",
                    task_description="This should fail"
                )
                self.fail("Should have raised ValueError for unknown agent")
            except ValueError as e:
                self.assertIn("Unknown agent", str(e))
            
            # Test activation without session
            orchestrator.conversation_state = None
            try:
                await orchestrator.activate_agent(
                    agent_id="ipdai_agent",
                    task_description="This should fail"
                )
                self.fail("Should have raised ValueError for no session")
            except ValueError as e:
                self.assertIn("No active conversation session", str(e))
        
        self.loop.run_until_complete(test_errors())
    
    def test_session_management(self):
        """Test session creation, loading, and management"""
        orchestrator = HAILEIOrchestrator(
            agents={},
            frameworks=self.frameworks,
            enable_logging=False
        )
        
        # Create multiple sessions
        session1 = orchestrator.create_session(self.course_request, "session_1")
        session2 = orchestrator.create_session(self.course_request, "session_2")
        
        # Verify sessions are tracked
        self.assertEqual(len(orchestrator.active_sessions), 2)
        self.assertIn("session_1", orchestrator.active_sessions)
        self.assertIn("session_2", orchestrator.active_sessions)
        
        # Test session loading
        loaded_session = orchestrator.load_session("session_1")
        self.assertEqual(loaded_session.session_id, "session_1")
        
        # Test non-existent session
        missing_session = orchestrator.load_session("nonexistent")
        self.assertIsNone(missing_session)


def run_end_to_end_tests():
    """Run all end-to-end tests and return results"""
    print("🧪 Running End-to-End Conversational Workflow Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test class
    tests = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndWorkflow)
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
            print(f"     {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
            print(f"     {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    run_end_to_end_tests()