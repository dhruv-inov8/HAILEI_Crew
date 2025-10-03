"""
Test Suite: Phase 3.1 - Parallel Execution System

Tests for parallel agent coordination, dependency analysis, and 
intelligent execution planning with performance optimization.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.parallel_orchestrator import (
    ParallelOrchestrator, 
    ParallelTask, 
    ExecutionMode, 
    AgentDependency,
    ExecutionPlan
)
from orchestrator.conversation_state import ConversationState, AgentOutput
from agents.agent_wrappers import AgentExecutionResult


class TestParallelTask(unittest.TestCase):
    """Test ParallelTask functionality"""
    
    def test_parallel_task_creation(self):
        """Test parallel task creation and serialization"""
        task = ParallelTask(
            agent_id="ipdai_agent",
            task_description="Create course foundation",
            context={"phase": "foundation_design"},
            dependencies=["coordinator_agent"],
            priority=3,
            estimated_duration=120.0
        )
        
        # Test properties
        self.assertEqual(task.agent_id, "ipdai_agent")
        self.assertEqual(task.priority, 3)
        self.assertEqual(task.estimated_duration, 120.0)
        self.assertEqual(len(task.dependencies), 1)
        
        # Test serialization
        task_dict = task.to_dict()
        self.assertIn('agent_id', task_dict)
        self.assertIn('dependencies', task_dict)
        self.assertIn('priority', task_dict)
        self.assertEqual(task_dict['agent_id'], "ipdai_agent")


class TestParallelOrchestrator(unittest.TestCase):
    """Test ParallelOrchestrator functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock main orchestrator
        self.mock_main_orchestrator = Mock()
        self.mock_main_orchestrator.activate_agent = AsyncMock()
        
        # Create parallel orchestrator
        self.parallel_orchestrator = ParallelOrchestrator(
            main_orchestrator=self.mock_main_orchestrator,
            max_parallel_agents=3,
            enable_logging=False
        )
        
        # Mock conversation state
        self.conversation_state = ConversationState(
            session_id="test_session",
            course_request={"course_title": "Test Course"}
        )
        self.conversation_state.set_current_phase("content_creation")
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_orchestrator_initialization(self):
        """Test parallel orchestrator initialization"""
        self.assertIsNotNone(self.parallel_orchestrator.agent_dependencies)
        self.assertIsNotNone(self.parallel_orchestrator.phase_execution_modes)
        self.assertEqual(self.parallel_orchestrator.max_parallel_agents, 3)
        
        # Verify dependency configuration
        self.assertIn('ipdai_agent', self.parallel_orchestrator.agent_dependencies)
        self.assertIn('cauthai_agent', self.parallel_orchestrator.agent_dependencies)
        
        # Verify phase modes
        self.assertEqual(
            self.parallel_orchestrator.phase_execution_modes['content_creation'],
            ExecutionMode.PARALLEL
        )
    
    def test_dependency_analysis(self):
        """Test agent dependency analysis"""
        # Create test tasks with dependencies
        tasks = [
            ParallelTask(
                agent_id="ipdai_agent",
                task_description="Create foundation",
                context={},
                dependencies=[],
                priority=3
            ),
            ParallelTask(
                agent_id="cauthai_agent", 
                task_description="Create content",
                context={},
                dependencies=["ipdai_agent"],
                priority=3
            ),
            ParallelTask(
                agent_id="ethosai_agent",
                task_description="Ethical review",
                context={},
                dependencies=[],
                priority=1
            )
        ]
        
        # Analyze dependencies
        execution_groups = self.parallel_orchestrator._analyze_dependencies(tasks)
        
        # Verify grouping
        self.assertIsInstance(execution_groups, list)
        self.assertGreater(len(execution_groups), 0)
        
        # IPDAi and EthosAi should be able to run in parallel (first group)
        # CAuthAi should run after IPDAi (second group)
        first_group = execution_groups[0]
        self.assertIn("ipdai_agent", first_group)
        self.assertIn("ethosai_agent", first_group)
    
    def test_execution_plan_creation(self):
        """Test execution plan creation"""
        tasks = [
            ParallelTask(
                agent_id="ipdai_agent",
                task_description="Foundation design",
                context={},
                dependencies=[],
                estimated_duration=60.0
            ),
            ParallelTask(
                agent_id="cauthai_agent",
                task_description="Content creation", 
                context={},
                dependencies=["ipdai_agent"],
                estimated_duration=90.0
            )
        ]
        
        # Create execution plan
        execution_plan = self.parallel_orchestrator._create_execution_plan("content_creation", tasks)
        
        # Verify plan
        self.assertIsInstance(execution_plan, ExecutionPlan)
        self.assertIsInstance(execution_plan.execution_mode, ExecutionMode)
        self.assertIsInstance(execution_plan.execution_groups, list)
        self.assertGreater(execution_plan.total_estimated_time, 0)
        self.assertGreaterEqual(execution_plan.parallel_efficiency, 0)
        
        # Test serialization
        plan_dict = execution_plan.to_dict()
        self.assertIn('execution_mode', plan_dict)
        self.assertIn('execution_groups', plan_dict)
    
    def test_parallel_time_calculation(self):
        """Test parallel execution time calculation"""
        tasks = [
            ParallelTask("agent1", "task1", {}, [], estimated_duration=60.0),
            ParallelTask("agent2", "task2", {}, [], estimated_duration=90.0),
            ParallelTask("agent3", "task3", {}, [], estimated_duration=30.0)
        ]
        
        # Test sequential groups (should sum)
        sequential_groups = [["agent1"], ["agent2"], ["agent3"]]
        sequential_time = self.parallel_orchestrator._calculate_parallel_time(sequential_groups, tasks)
        self.assertEqual(sequential_time, 180.0)  # 60 + 90 + 30
        
        # Test parallel group (should take max)
        parallel_groups = [["agent1", "agent2", "agent3"]]
        parallel_time = self.parallel_orchestrator._calculate_parallel_time(parallel_groups, tasks)
        self.assertEqual(parallel_time, 90.0)  # max(60, 90, 30)
    
    def test_single_agent_execution(self):
        """Test single agent execution"""
        # Mock agent output
        mock_output = AgentOutput(
            agent_id="ipdai_agent",
            agent_name="IPDAi",
            phase="foundation_design",
            content="Foundation created successfully",
            timestamp=datetime.now()
        )
        self.mock_main_orchestrator.activate_agent.return_value = mock_output
        
        async def test_execution():
            task = ParallelTask(
                agent_id="ipdai_agent",
                task_description="Create foundation",
                context={"phase": "foundation_design"},
                dependencies=[]
            )
            
            result = await self.parallel_orchestrator._execute_single_agent(
                "ipdai_agent", task, self.conversation_state
            )
            
            # Verify execution
            self.assertEqual(result.agent_id, "ipdai_agent")
            self.assertEqual(result.content, "Foundation created successfully")
            self.mock_main_orchestrator.activate_agent.assert_called_once()
        
        self.loop.run_until_complete(test_execution())
    
    def test_parallel_group_execution(self):
        """Test parallel group execution"""
        # Mock multiple agent outputs
        mock_outputs = {
            "ethosai_agent": AgentOutput(
                agent_id="ethosai_agent",
                agent_name="EthosAi",
                phase="foundation_design",
                content="Ethical review completed",
                timestamp=datetime.now()
            ),
            "searchai_agent": AgentOutput(
                agent_id="searchai_agent", 
                agent_name="SearchAi",
                phase="foundation_design",
                content="Resources gathered",
                timestamp=datetime.now()
            )
        }
        
        def mock_activate_agent(agent_id, **kwargs):
            return mock_outputs[agent_id]
        
        self.mock_main_orchestrator.activate_agent.side_effect = mock_activate_agent
        
        async def test_parallel():
            task_map = {
                "ethosai_agent": ParallelTask(
                    agent_id="ethosai_agent",
                    task_description="Ethical review",
                    context={},
                    dependencies=[]
                ),
                "searchai_agent": ParallelTask(
                    agent_id="searchai_agent",
                    task_description="Resource search",
                    context={},
                    dependencies=[]
                )
            }
            
            results = await self.parallel_orchestrator._execute_agent_group(
                ["ethosai_agent", "searchai_agent"],
                task_map,
                self.conversation_state
            )
            
            # Verify parallel execution
            self.assertEqual(len(results), 2)
            self.assertIn("ethosai_agent", results)
            self.assertIn("searchai_agent", results)
            self.assertEqual(results["ethosai_agent"].content, "Ethical review completed")
            self.assertEqual(results["searchai_agent"].content, "Resources gathered")
        
        self.loop.run_until_complete(test_parallel())
    
    def test_error_handling_in_execution(self):
        """Test error handling during agent execution"""
        # Mock agent failure
        self.mock_main_orchestrator.activate_agent.side_effect = Exception("Agent failed")
        
        async def test_error():
            task = ParallelTask(
                agent_id="failing_agent",
                task_description="This will fail",
                context={},
                dependencies=[]
            )
            
            result = await self.parallel_orchestrator._execute_single_agent(
                "failing_agent", task, self.conversation_state
            )
            
            # Verify error handling
            self.assertEqual(result.agent_id, "failing_agent")
            self.assertIn("Agent execution failed", result.content)
            self.assertTrue(result.metadata.get('error', False))
        
        self.loop.run_until_complete(test_error())
    
    def test_performance_tracking(self):
        """Test execution performance tracking"""
        # Create mock execution plan
        execution_plan = ExecutionPlan(
            execution_mode=ExecutionMode.PARALLEL,
            execution_groups=[["agent1", "agent2"]],
            total_estimated_time=120.0,
            parallel_efficiency=0.5
        )
        
        # Mock results
        results = {
            "agent1": AgentOutput("agent1", "Agent1", "test", "Result1", datetime.now()),
            "agent2": AgentOutput("agent2", "Agent2", "test", "Result2", datetime.now())
        }
        
        # Track performance
        self.parallel_orchestrator._track_execution_performance(
            "test_phase", execution_plan, 100.0, results
        )
        
        # Verify tracking
        self.assertEqual(len(self.parallel_orchestrator.execution_history), 1)
        
        history_entry = self.parallel_orchestrator.execution_history[0]
        self.assertEqual(history_entry['phase_id'], "test_phase")
        self.assertEqual(history_entry['actual_time'], 100.0)
        self.assertEqual(history_entry['estimated_time'], 120.0)
        self.assertEqual(history_entry['successful_agents'], 2)
    
    def test_execution_statistics(self):
        """Test execution statistics generation"""
        # Add some mock performance data
        self.parallel_orchestrator.execution_history = [
            {
                'phase_id': 'test1',
                'execution_mode': 'parallel',
                'actual_time': 100.0,
                'estimated_time': 120.0,
                'parallel_efficiency': 0.4,
                'successful_agents': 2,
                'total_agents': 2
            },
            {
                'phase_id': 'test2', 
                'execution_mode': 'sequential',
                'actual_time': 150.0,
                'estimated_time': 140.0,
                'parallel_efficiency': 0.0,
                'successful_agents': 3,
                'total_agents': 3
            }
        ]
        
        self.parallel_orchestrator.parallel_efficiency_stats = {
            'test1': 0.4,
            'test2': 0.0
        }
        
        # Get statistics
        stats = self.parallel_orchestrator.get_execution_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_executions'], 2)
        self.assertEqual(stats['average_parallel_efficiency'], 0.2)  # (0.4 + 0.0) / 2
        self.assertIn('execution_mode_distribution', stats)
        self.assertIn('phase_efficiencies', stats)
        self.assertIn('recent_executions', stats)


class TestIntegrationWithMainOrchestrator(unittest.TestCase):
    """Test integration between parallel and main orchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    @patch('orchestrator.conversation_orchestrator.ConversationalAgentFactory')
    def test_parallel_execution_integration(self, mock_factory):
        """Test parallel execution integration with main orchestrator"""
        from orchestrator import HAILEIOrchestrator
        
        # Mock the factory to avoid agent creation
        mock_factory.return_value.create_all_agents.return_value = {}
        
        # Create orchestrator
        orchestrator = HAILEIOrchestrator(
            agents={},
            frameworks={},
            enable_logging=False
        )
        
        # Verify parallel orchestrator is initialized
        self.assertIsNotNone(orchestrator.parallel_orchestrator)
        self.assertIsInstance(orchestrator.parallel_orchestrator, ParallelOrchestrator)
        
        # Test default phase tasks creation
        foundation_tasks = orchestrator._create_default_phase_tasks('foundation_design')
        self.assertGreater(len(foundation_tasks), 0)
        
        # Verify task structure
        for task in foundation_tasks:
            self.assertIn('agent_id', task)
            self.assertIn('task_description', task)
            self.assertIn('dependencies', task)
            self.assertIn('priority', task)
    
    def test_execution_mode_configuration(self):
        """Test execution mode configuration for different phases"""
        mock_orchestrator = Mock()
        parallel_orchestrator = ParallelOrchestrator(mock_orchestrator, enable_logging=False)
        
        # Test phase mode mappings
        self.assertEqual(
            parallel_orchestrator.phase_execution_modes['course_overview'],
            ExecutionMode.SEQUENTIAL
        )
        self.assertEqual(
            parallel_orchestrator.phase_execution_modes['content_creation'],
            ExecutionMode.PARALLEL
        )
        self.assertEqual(
            parallel_orchestrator.phase_execution_modes['final_integration'],
            ExecutionMode.SEQUENTIAL
        )


def run_phase_3_1_tests():
    """Run all Phase 3.1 tests and return results"""
    print("🧪 Running Phase 3.1 Tests: Parallel Execution System")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestParallelTask,
        TestParallelOrchestrator,
        TestIntegrationWithMainOrchestrator
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
    run_phase_3_1_tests()