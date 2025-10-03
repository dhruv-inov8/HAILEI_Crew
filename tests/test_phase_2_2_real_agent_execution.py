"""
Test Suite: Phase 2.2 - Real Agent Task Execution

Tests for actual agent execution integration with conversational orchestrator.
Validates real CrewAI agent execution through wrapper system.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import HAILEIOrchestrator, ConversationState
from agents.conversational_agents import ConversationalAgentFactory
from agents.agent_wrappers import AgentExecutor


class TestRealAgentExecution(unittest.TestCase):
    """Test real agent execution integration"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock frameworks
        self.frameworks = {
            'kdka': {'summary': 'Knowledge, Delivery, Context, Assessment framework'},
            'prrr': {'summary': 'Personal, Relatable, Relative, Real-world framework'}
        }
        
        # Sample course request
        self.course_request = {
            'course_title': 'Test AI Course',
            'course_description': 'Introduction to AI concepts',
            'course_level': 'Undergraduate',
            'course_duration_weeks': 16
        }
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_orchestrator_initialization_with_real_agents(self):
        """Test orchestrator initializes with real agent system"""
        try:
            # Create orchestrator with minimal agents dict (can be empty for this test)
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks=self.frameworks,
                enable_logging=False
            )
            
            # Verify components are initialized
            self.assertIsNotNone(orchestrator.agent_factory)
            self.assertIsNotNone(orchestrator.agent_executor)
            self.assertIsInstance(orchestrator.conversational_agents, dict)
            
            # Verify agent factory is working
            self.assertIsInstance(orchestrator.agent_factory, ConversationalAgentFactory)
            self.assertIsInstance(orchestrator.agent_executor, AgentExecutor)
            
        except Exception as e:
            self.fail(f"Orchestrator initialization failed: {e}")
    
    def test_agent_id_mapping(self):
        """Test agent ID mapping from CrewAI agents"""
        orchestrator = HAILEIOrchestrator(
            agents={},
            frameworks=self.frameworks,
            enable_logging=False
        )
        
        # Test mock agents with different roles
        mock_coordinator = Mock()
        mock_coordinator.role = "HAILEI Educational Intelligence Coordinator"
        
        mock_ipdai = Mock()
        mock_ipdai.role = "Instructional Planning & Design Agent"
        
        mock_cauthai = Mock()
        mock_cauthai.role = "Content Authoring Agent"
        
        # Test agent ID mapping
        coord_id = orchestrator._get_agent_id_from_agent(mock_coordinator)
        ipdai_id = orchestrator._get_agent_id_from_agent(mock_ipdai)
        cauthai_id = orchestrator._get_agent_id_from_agent(mock_cauthai)
        
        self.assertEqual(coord_id, "hailei4t_coordinator_agent")
        self.assertEqual(ipdai_id, "ipdai_agent")
        self.assertEqual(cauthai_id, "cauthai_agent")
    
    def test_session_creation_with_real_agents(self):
        """Test session creation works with real agent system"""
        orchestrator = HAILEIOrchestrator(
            agents={},
            frameworks=self.frameworks,
            enable_logging=False
        )
        
        # Create session
        session = orchestrator.create_session(self.course_request)
        
        # Verify session created
        self.assertIsNotNone(session)
        self.assertIsInstance(session, ConversationState)
        self.assertEqual(session.course_request, self.course_request)
        self.assertEqual(session.frameworks, self.frameworks)
        
        # Verify session is tracked
        self.assertIn(session.session_id, orchestrator.active_sessions)
    
    @patch('agents.agent_wrappers.ConversationalAgentWrapper.execute_task')
    def test_real_agent_execution_flow(self, mock_execute):
        """Test the flow of real agent execution"""
        # Mock successful agent execution
        from agents.agent_wrappers import AgentExecutionResult
        mock_execute.return_value = AgentExecutionResult(
            agent_id="ipdai_agent",
            task_description="Test task",
            output="Test agent output for course foundation design",
            execution_time=1.2,
            success=True
        )
        
        async def test_execution():
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks=self.frameworks,
                enable_logging=False
            )
            
            # Create mock agent that maps to ipdai_agent
            mock_agent = Mock()
            mock_agent.role = "Instructional Planning & Design Agent"
            
            # Create session
            session = orchestrator.create_session(self.course_request)
            orchestrator.conversation_state = session
            
            # Mock that we have the conversational agent
            mock_wrapper = Mock()
            orchestrator.conversational_agents["ipdai_agent"] = mock_wrapper
            
            # Prepare context
            context = {
                'course_request': self.course_request,
                'current_phase': 'foundation_design',
                'frameworks': self.frameworks,
                'previous_outputs': {}
            }
            
            # Execute agent task
            result = await orchestrator._execute_agent_task(
                mock_agent,
                "Create course foundation using KDKA framework",
                context
            )
            
            # Verify execution happened and returned result
            self.assertIn("Test agent output", result)
        
        self.loop.run_until_complete(test_execution())
    
    def test_fallback_execution(self):
        """Test fallback execution for unmapped agents"""
        async def test_fallback():
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks=self.frameworks,
                enable_logging=False
            )
            
            # Create mock agent that doesn't map to conversational system
            mock_agent = Mock()
            mock_agent.role = "Unknown Agent Role"
            
            # Prepare context
            context = {
                'course_request': self.course_request,
                'current_phase': 'test_phase',
                'frameworks': self.frameworks
            }
            
            # Execute should fall back to placeholder
            result = await orchestrator._execute_agent_task(
                mock_agent,
                "Test task",
                context
            )
            
            # Verify fallback execution
            self.assertIn("Unknown Agent Role", result)
            self.assertIn("Test task", result)
            self.assertIn("fallback execution", result)
        
        self.loop.run_until_complete(test_fallback())
    
    def test_conversation_context_preparation(self):
        """Test conversation context is properly prepared for agents"""
        orchestrator = HAILEIOrchestrator(
            agents={},
            frameworks=self.frameworks,
            enable_logging=False
        )
        
        # Create session with some history
        session = orchestrator.create_session(self.course_request)
        session.add_conversation_turn("user", "I want to create an AI course")
        session.add_conversation_turn("coordinator", "Let's start with foundation design")
        orchestrator.conversation_state = session
        
        # Prepare context
        context = orchestrator._prepare_agent_context("ipdai_agent", {"test": "additional"})
        
        # Verify context includes all necessary information
        self.assertIn('course_request', context)
        self.assertIn('frameworks', context)
        self.assertIn('conversation_history', context)
        self.assertIn('test', context)  # Additional context
        
        # Verify conversation history is included
        self.assertEqual(len(context['conversation_history']), 2)
        self.assertEqual(context['conversation_history'][0]['speaker'], 'user')


class TestAgentExecutorIntegration(unittest.TestCase):
    """Test integration between orchestrator and agent executor"""
    
    def setUp(self):
        """Set up test environment"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    @patch('agents.conversational_agents.ConversationalAgentFactory.create_all_agents')
    def test_agent_registration_in_executor(self, mock_create_agents):
        """Test that agents are properly registered in executor"""
        # Mock agent creation
        mock_wrapper = Mock()
        mock_wrapper.agent_id = "test_agent"
        mock_create_agents.return_value = {"test_agent": mock_wrapper}
        
        orchestrator = HAILEIOrchestrator(
            agents={},
            frameworks={},
            enable_logging=False
        )
        
        # Verify agent was registered
        self.assertIn("test_agent", orchestrator.conversational_agents)
        self.assertEqual(orchestrator.conversational_agents["test_agent"], mock_wrapper)
    
    def test_error_handling_in_agent_initialization(self):
        """Test error handling when agent initialization fails"""
        with patch('agents.conversational_agents.ConversationalAgentFactory.create_all_agents', 
                   side_effect=Exception("Agent creation failed")):
            
            # Should not crash, should fall back gracefully
            orchestrator = HAILEIOrchestrator(
                agents={},
                frameworks={},
                enable_logging=False
            )
            
            # Should have empty conversational agents
            self.assertEqual(len(orchestrator.conversational_agents), 0)


def run_phase_2_2_tests():
    """Run all Phase 2.2 tests and return results"""
    print("🧪 Running Phase 2.2 Tests: Real Agent Task Execution")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRealAgentExecution,
        TestAgentExecutorIntegration
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
    run_phase_2_2_tests()