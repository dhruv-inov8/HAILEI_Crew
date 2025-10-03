"""
Test Suite: Phase 2.1 - Agent Wrapper System

Tests for ConversationalAgentWrapper, AgentExecutor, and ConversationalAgentFactory.
Validates agent integration and execution with conversation context.
"""

import unittest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_wrappers import (
    ConversationalAgentWrapper,
    AgentExecutor,
    AgentExecutionResult
)
from agents.conversational_agents import ConversationalAgentFactory


class TestAgentExecutionResult(unittest.TestCase):
    """Test AgentExecutionResult functionality"""
    
    def test_execution_result_creation(self):
        """Test execution result creation and serialization"""
        result = AgentExecutionResult(
            agent_id="test_agent",
            task_description="Test task for agent",
            output="Agent completed the test task successfully",
            execution_time=1.5,
            success=True,
            metadata={"test_key": "test_value"}
        )
        
        # Test properties
        self.assertEqual(result.agent_id, "test_agent")
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 1.5)
        self.assertIsNone(result.error_message)
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIn('agent_id', result_dict)
        self.assertIn('execution_time', result_dict)
        self.assertIn('success', result_dict)
        self.assertIn('metadata', result_dict)
        
        # Verify JSON serializable
        json_str = json.dumps(result_dict)
        self.assertIsInstance(json_str, str)
    
    def test_error_result_creation(self):
        """Test error result creation"""
        error_result = AgentExecutionResult(
            agent_id="failing_agent",
            task_description="Task that will fail",
            output="",
            execution_time=0.5,
            success=False,
            error_message="Mock execution error"
        )
        
        # Test error properties
        self.assertFalse(error_result.success)
        self.assertEqual(error_result.error_message, "Mock execution error")
        self.assertEqual(error_result.output, "")


class TestConversationalAgentWrapper(unittest.TestCase):
    """Test ConversationalAgentWrapper functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock CrewAI agent
        self.mock_crewai_agent = Mock()
        self.mock_crewai_agent.role = "Test Agent"
        self.mock_crewai_agent.execute_task = Mock(return_value="Mock agent output")
        
        # Create wrapper
        self.wrapper = ConversationalAgentWrapper(
            agent_id="test_agent",
            crewai_agent=self.mock_crewai_agent,
            timeout_seconds=30  # Short timeout for testing
        )
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization"""
        self.assertEqual(self.wrapper.agent_id, "test_agent")
        self.assertEqual(self.wrapper.crewai_agent, self.mock_crewai_agent)
        self.assertEqual(self.wrapper.execution_count, 0)
        self.assertEqual(len(self.wrapper.conversation_memory), 0)
        
        # Test performance metrics initialization
        metrics = self.wrapper.get_performance_metrics()
        self.assertEqual(metrics['total_executions'], 0)
        self.assertEqual(metrics['successful_executions'], 0)
        self.assertEqual(metrics['agent_id'], "test_agent")
    
    @patch('agents.agent_wrappers.Task')
    def test_context_preparation(self, mock_task_class):
        """Test context preparation for agent tasks"""
        # Mock Task creation
        mock_task = Mock()
        mock_task.description = "Mocked task with Create course foundation, Test Course, foundation_design, Previous work done"
        mock_task_class.return_value = mock_task
        
        # Sample context
        context = {
            'course_request': {
                'course_title': 'Test Course',
                'course_level': 'Beginner'
            },
            'frameworks': {
                'kdka': {'summary': 'KDKA framework'},
                'prrr': {'summary': 'PRRR framework'}
            },
            'current_phase': 'foundation_design',
            'previous_outputs': {
                'other_agent': {
                    'content': 'Previous work done',
                    'phase': 'previous_phase'
                }
            }
        }
        
        conversation_history = [
            {'speaker': 'user', 'message': 'I want to create a course'},
            {'speaker': 'coordinator', 'message': 'Let\'s start with foundation design'}
        ]
        
        # Prepare task with context
        task = self.wrapper._prepare_contextual_task(
            "Create course foundation",
            context,
            conversation_history
        )
        
        # Verify task creation
        self.assertIsNotNone(task)
        self.assertIn("Create course foundation", task.description)
        self.assertIn("Test Course", task.description)  # Course title included
        self.assertIn("foundation_design", task.description)  # Current phase included
        self.assertIn("Previous work done", task.description)  # Previous outputs included
    
    def test_context_formatting(self):
        """Test context section formatting"""
        context = {
            'course_request': {
                'course_title': 'Advanced AI Course',
                'course_level': 'Graduate',
                'course_duration_weeks': 16
            },
            'frameworks': {
                'kdka': {'summary': 'Knowledge framework'},
                'prrr': {'summary': 'Engagement framework'}
            },
            'current_phase': 'content_creation'
        }
        
        formatted_context = self.wrapper._format_context_section(context)
        
        # Verify context formatting
        self.assertIn("COURSE CONTEXT:", formatted_context)
        self.assertIn("Advanced AI Course", formatted_context)
        self.assertIn("Graduate", formatted_context)
        self.assertIn("16 weeks", formatted_context)
        self.assertIn("KDKA FRAMEWORK", formatted_context)
        self.assertIn("PRRR FRAMEWORK", formatted_context)
        self.assertIn("CURRENT PHASE: content_creation", formatted_context)
    
    def test_conversation_history_formatting(self):
        """Test conversation history formatting"""
        history = [
            {'speaker': 'user', 'message': 'Hello, I want to create a machine learning course'},
            {'speaker': 'coordinator', 'message': 'Great! Let\'s start with the course foundation'},
            {'speaker': 'ipdai', 'message': 'I\'ll create the learning objectives using KDKA framework'}
        ]
        
        formatted_history = self.wrapper._format_conversation_history(history)
        
        # Verify history formatting
        self.assertIn("RECENT CONVERSATION:", formatted_history)
        self.assertIn("user:", formatted_history)
        self.assertIn("coordinator:", formatted_history)
        self.assertIn("ipdai:", formatted_history)
    
    def test_memory_management(self):
        """Test agent memory functionality"""
        # Initially empty memory
        memory = self.wrapper.get_conversation_memory()
        self.assertEqual(len(memory), 0)
        
        # Update memory
        self.wrapper.update_memory("course_focus", "Machine Learning")
        self.wrapper.update_memory("framework_preference", "KDKA")
        
        # Verify memory updated
        memory = self.wrapper.get_conversation_memory()
        self.assertEqual(memory["course_focus"], "Machine Learning")
        self.assertEqual(memory["framework_preference"], "KDKA")
        
        # Test memory formatting
        memory_formatted = self.wrapper._format_agent_memory()
        self.assertIn("AGENT MEMORY:", memory_formatted)
        self.assertIn("course_focus", memory_formatted)
        self.assertIn("framework_preference", memory_formatted)
        
        # Clear memory
        self.wrapper.clear_memory()
        memory = self.wrapper.get_conversation_memory()
        self.assertEqual(len(memory), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        # Initial metrics
        metrics = self.wrapper.get_performance_metrics()
        self.assertEqual(metrics['total_executions'], 0)
        self.assertEqual(metrics['success_rate'], 0.0)
        
        # Simulate successful execution
        self.wrapper._update_performance_metrics(1.5, True)
        
        # Check updated metrics
        metrics = self.wrapper.get_performance_metrics()
        self.assertEqual(metrics['total_executions'], 1)
        self.assertEqual(metrics['successful_executions'], 1)
        self.assertEqual(metrics['failed_executions'], 0)
        self.assertEqual(metrics['success_rate'], 1.0)
        self.assertEqual(metrics['average_execution_time'], 1.5)
        
        # Simulate failed execution
        self.wrapper._update_performance_metrics(0.8, False)
        
        # Check metrics after failure
        metrics = self.wrapper.get_performance_metrics()
        self.assertEqual(metrics['total_executions'], 2)
        self.assertEqual(metrics['successful_executions'], 1)
        self.assertEqual(metrics['failed_executions'], 1)
        self.assertEqual(metrics['success_rate'], 0.5)
        self.assertEqual(metrics['average_execution_time'], 1.5)  # Only successful executions
    
    def test_memory_update_after_execution(self):
        """Test memory update after task execution"""
        # Update conversation memory
        self.wrapper._update_conversation_memory(
            "Create learning objectives",
            "Generated 5 learning objectives using Bloom's taxonomy",
            {"current_phase": "foundation_design"}
        )
        
        # Verify memory updated
        memory = self.wrapper.get_conversation_memory()
        self.assertIn('recent_executions', memory)
        self.assertIn('total_executions', memory)
        self.assertIn('last_execution', memory)
        
        # Check recent execution
        recent = memory['recent_executions'][0]
        self.assertIn('task', recent)
        self.assertIn('result_summary', recent)
        self.assertIn('context_phase', recent)
        self.assertEqual(recent['context_phase'], 'foundation_design')


class TestAgentExecutor(unittest.TestCase):
    """Test AgentExecutor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.executor = AgentExecutor(
            max_concurrent_executions=2
        )
        
        # Create mock agent wrappers
        self.mock_wrapper1 = Mock()
        self.mock_wrapper1.agent_id = "agent1"
        self.mock_wrapper1.execute_task = AsyncMock(return_value=AgentExecutionResult(
            agent_id="agent1",
            task_description="Test task 1",
            output="Agent 1 output",
            execution_time=1.0,
            success=True
        ))
        self.mock_wrapper1.get_conversation_memory.return_value = {"test": "memory"}
        self.mock_wrapper1.get_performance_metrics.return_value = {
            "total_executions": 1,
            "successful_executions": 1,
            "average_execution_time": 1.0
        }
        
        self.mock_wrapper2 = Mock()
        self.mock_wrapper2.agent_id = "agent2" 
        self.mock_wrapper2.execute_task = AsyncMock(return_value=AgentExecutionResult(
            agent_id="agent2",
            task_description="Test task 2",
            output="Agent 2 output",
            execution_time=1.5,
            success=True
        ))
        self.mock_wrapper2.get_conversation_memory.return_value = {"test": "memory"}
        self.mock_wrapper2.get_performance_metrics.return_value = {
            "total_executions": 1,
            "successful_executions": 1,
            "average_execution_time": 1.5
        }
        
        # Register mock wrappers
        self.executor.register_agent(self.mock_wrapper1)
        self.executor.register_agent(self.mock_wrapper2)
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_agent_registration(self):
        """Test agent registration and management"""
        # Verify agents registered
        self.assertIn("agent1", self.executor.agents)
        self.assertIn("agent2", self.executor.agents)
        
        # Test system status
        status = self.executor.get_system_status()
        self.assertEqual(len(status['registered_agents']), 2)
        self.assertEqual(status['max_concurrent_executions'], 2)
        self.assertEqual(status['total_agents'], 2)
        
        # Test agent unregistration
        self.executor.unregister_agent("agent1")
        self.assertNotIn("agent1", self.executor.agents)
        self.assertIn("agent2", self.executor.agents)
    
    def test_single_agent_execution(self):
        """Test single agent execution"""
        async def test_execution():
            result = await self.executor.execute_agent_task(
                agent_id="agent1",
                task_description="Test single execution",
                context={"test": "context"}
            )
            
            # Verify execution result
            self.assertEqual(result.agent_id, "agent1")
            self.assertTrue(result.success)
            self.assertEqual(result.output, "Agent 1 output")
            
            # Verify agent wrapper was called
            self.mock_wrapper1.execute_task.assert_called_once()
        
        self.loop.run_until_complete(test_execution())
    
    def test_parallel_agent_execution(self):
        """Test parallel agent execution"""
        async def test_parallel():
            agent_tasks = [
                {
                    'agent_id': 'agent1',
                    'task_description': 'Parallel task 1',
                    'context': {'phase': 'foundation'}
                },
                {
                    'agent_id': 'agent2',
                    'task_description': 'Parallel task 2',
                    'context': {'phase': 'content'}
                }
            ]
            
            results = await self.executor.execute_multiple_agents(agent_tasks)
            
            # Verify both agents executed
            self.assertEqual(len(results), 2)
            self.assertIn('agent1', results)
            self.assertIn('agent2', results)
            
            # Verify execution results
            self.assertTrue(results['agent1'].success)
            self.assertTrue(results['agent2'].success)
            
            # Verify both wrappers were called
            self.mock_wrapper1.execute_task.assert_called()
            self.mock_wrapper2.execute_task.assert_called()
        
        self.loop.run_until_complete(test_parallel())
    
    def test_agent_status_tracking(self):
        """Test agent status tracking"""
        # Get status for specific agent
        agent1_status = self.executor.get_agent_status("agent1")
        self.assertEqual(agent1_status['agent_id'], "agent1")
        self.assertIn('performance_metrics', agent1_status)
        self.assertIn('is_active', agent1_status)
        
        # Get status for all agents
        all_status = self.executor.get_agent_status()
        self.assertEqual(len(all_status), 2)
        self.assertIn('agent1', all_status)
        self.assertIn('agent2', all_status)
    
    def test_error_handling(self):
        """Test error handling in agent execution"""
        async def test_error():
            # Try to execute non-existent agent
            try:
                await self.executor.execute_agent_task(
                    agent_id="nonexistent_agent",
                    task_description="This will fail"
                )
                self.fail("Should have raised ValueError")
            except ValueError as e:
                self.assertIn("not registered", str(e))
        
        self.loop.run_until_complete(test_error())


class TestConversationalAgentFactory(unittest.TestCase):
    """Test ConversationalAgentFactory functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock agent configuration file
        self.mock_config = {
            'test_agent_1': {
                'role': 'Test Agent 1',
                'goal': 'Complete test tasks',
                'backstory': 'A test agent for validation purposes',
                'allow_delegation': False
            },
            'test_agent_2': {
                'role': 'Test Agent 2', 
                'goal': 'Handle secondary test tasks',
                'backstory': 'Another test agent for comprehensive testing',
                'allow_delegation': True
            }
        }
        
        # Mock file operations
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=self.mock_config), \
             patch('os.path.abspath', return_value='/mock/path/config/agents.yaml'):
            
            self.factory = ConversationalAgentFactory(enable_logging=False)
    
    def test_factory_initialization(self):
        """Test factory initialization"""
        # Verify configurations loaded
        self.assertEqual(len(self.factory.agent_configs), 2)
        self.assertIn('test_agent_1', self.factory.agent_configs)
        self.assertIn('test_agent_2', self.factory.agent_configs)
        
        # Verify agent list
        agent_list = self.factory.list_agents()
        self.assertEqual(len(agent_list), 2)
        self.assertIn('test_agent_1', agent_list)
        self.assertIn('test_agent_2', agent_list)
    
    def test_tool_preparation(self):
        """Test tool preparation for agents"""
        # Test tool assignment for different agent types
        ipdai_tools = self.factory._prepare_agent_tools('ipdai_agent', {})
        self.assertEqual(len(ipdai_tools), 1)  # Should have blooms_taxonomy_tool
        
        cauthai_tools = self.factory._prepare_agent_tools('cauthai_agent', {})
        self.assertEqual(len(cauthai_tools), 1)  # Should have resource_search_tool
        
        editorai_tools = self.factory._prepare_agent_tools('editorai_agent', {})
        self.assertEqual(len(editorai_tools), 2)  # Should have accessibility and blooms tools
        
        coordinator_tools = self.factory._prepare_agent_tools('hailei4t_coordinator_agent', {})
        self.assertEqual(len(coordinator_tools), 0)  # Coordinator has no tools
    
    @patch('agents.conversational_agents.Agent')
    def test_agent_creation(self, mock_agent_class):
        """Test individual agent creation"""
        # Mock CrewAI Agent creation
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        # Create agent
        wrapper = self.factory.create_agent('test_agent_1')
        
        # Verify wrapper created
        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.agent_id, 'test_agent_1')
        
        # Verify CrewAI agent was created with correct parameters
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs
        self.assertEqual(call_kwargs['role'], 'Test Agent 1')
        self.assertEqual(call_kwargs['goal'], 'Complete test tasks')
        self.assertFalse(call_kwargs['allow_delegation'])
    
    def test_agent_info_retrieval(self):
        """Test agent information retrieval"""
        # Get info for existing agent
        info = self.factory.get_agent_info('test_agent_1')
        self.assertIsNotNone(info)
        self.assertEqual(info['agent_id'], 'test_agent_1')
        self.assertEqual(info['role'], 'Test Agent 1')
        self.assertFalse(info['allow_delegation'])
        self.assertFalse(info['has_wrapper'])  # No wrapper created yet
        
        # Get info for non-existent agent
        info = self.factory.get_agent_info('nonexistent_agent')
        self.assertIsNone(info)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        validation_results = self.factory.validate_configurations()
        
        # Verify validation structure
        self.assertIn('valid_agents', validation_results)
        self.assertIn('invalid_agents', validation_results)
        self.assertIn('warnings', validation_results)
        
        # Both test agents should be valid (they have all required fields)
        self.assertEqual(len(validation_results['valid_agents']), 2)
        self.assertEqual(len(validation_results['invalid_agents']), 0)
    
    def test_specialized_agents_mapping(self):
        """Test specialized agents dictionary creation"""
        # Mock agent creation
        with patch.object(self.factory, 'create_all_agents'), \
             patch.object(self.factory, 'conversational_wrappers') as mock_wrappers:
            
            # Mock wrapper instances
            mock_wrappers.__contains__ = lambda self, key: key in [
                'hailei4t_coordinator_agent', 'ipdai_agent', 'cauthai_agent'
            ]
            mock_wrappers.__getitem__ = lambda self, key: Mock(agent_id=key)
            
            specialized = self.factory.create_specialized_agents_dict()
            
            # Verify mapping created
            self.assertIn('coordinator', specialized)
            self.assertIn('ipdai', specialized)
            self.assertIn('cauthai', specialized)


def run_phase_3_tests():
    """Run all Phase 3 tests and return results"""
    print("🧪 Running Phase 2.1 Tests: Agent Wrapper System")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAgentExecutionResult,
        TestConversationalAgentWrapper,
        TestAgentExecutor,
        TestConversationalAgentFactory
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
    run_phase_3_tests()