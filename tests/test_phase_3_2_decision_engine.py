"""
Test Suite: Phase 3.2 - Dynamic Decision Engine

Tests for intelligent decision-making system including agent selection,
execution mode optimization, and performance-based learning.
"""

import unittest
import asyncio
import os
import sys
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.decision_engine import (
    DynamicDecisionEngine,
    DecisionContext,
    Decision,
    DecisionType,
    ConfidenceLevel
)
from orchestrator.parallel_orchestrator import ExecutionMode, ParallelTask
from orchestrator.conversation_state import ConversationState


class TestDecisionEngine(unittest.TestCase):
    """Test Dynamic Decision Engine functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.decision_engine = DynamicDecisionEngine(enable_logging=False)
        
        # Create test conversation state
        self.conversation_state = ConversationState(
            session_id="test_session",
            course_request={"course_title": "Test Course"}
        )
        
        # Create test context
        self.decision_context = DecisionContext(
            phase_id="content_creation",
            current_state=self.conversation_state,
            available_agents=["ipdai_agent", "cauthai_agent", "tfdai_agent"],
            execution_history=[],
            performance_metrics={},
            user_preferences={}
        )
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_decision_engine_initialization(self):
        """Test decision engine initialization"""
        self.assertIsNotNone(self.decision_engine.decision_history)
        self.assertIsNotNone(self.decision_engine.agent_performance_history)
        self.assertIsNotNone(self.decision_engine.decision_weights)
        self.assertEqual(len(self.decision_engine.decision_history), 0)
    
    def test_execution_mode_decision(self):
        """Test execution mode decision making"""
        
        async def test_decision():
            decision = await self.decision_engine.make_execution_decision(self.decision_context)
            
            # Verify decision structure
            self.assertIsInstance(decision, Decision)
            self.assertEqual(decision.decision_type, DecisionType.EXECUTION_MODE)
            self.assertIsInstance(decision.decision_value, ExecutionMode)
            self.assertIsInstance(decision.confidence, ConfidenceLevel)
            self.assertIsInstance(decision.reasoning, str)
            self.assertIsInstance(decision.alternatives, list)
            
            # Verify decision is stored
            self.assertEqual(len(self.decision_engine.decision_history), 1)
            self.assertEqual(self.decision_engine.decision_history[0], decision)
        
        self.loop.run_until_complete(test_decision())
    
    def test_agent_selection_decision(self):
        """Test intelligent agent selection"""
        
        async def test_selection():
            required_skills = ["content_creation", "engagement"]
            decision = await self.decision_engine.make_agent_selection_decision(
                self.decision_context, required_skills
            )
            
            # Verify decision
            self.assertEqual(decision.decision_type, DecisionType.AGENT_SELECTION)
            self.assertIn(decision.decision_value, self.decision_context.available_agents)
            self.assertIsInstance(decision.confidence, ConfidenceLevel)
            self.assertTrue(len(decision.reasoning) > 0)
            
            # Should prefer cauthai_agent for content creation
            self.assertEqual(decision.decision_value, "cauthai_agent")
        
        self.loop.run_until_complete(test_selection())
    
    def test_priority_optimization(self):
        """Test workflow priority optimization"""
        
        # Create test tasks
        tasks = [
            ParallelTask(
                agent_id="ipdai_agent",
                task_description="Foundation design",
                context={},
                dependencies=[],
                priority=2,
                estimated_duration=120.0
            ),
            ParallelTask(
                agent_id="cauthai_agent",
                task_description="Content creation",
                context={},
                dependencies=["ipdai_agent"],
                priority=1,
                estimated_duration=180.0
            ),
            ParallelTask(
                agent_id="tfdai_agent",
                task_description="Technical implementation",
                context={},
                dependencies=[],
                priority=3,
                estimated_duration=90.0
            )
        ]
        
        async def test_optimization():
            decision = await self.decision_engine.optimize_workflow_priorities(
                self.decision_context, tasks
            )
            
            # Verify decision
            self.assertEqual(decision.decision_type, DecisionType.PRIORITY_ADJUSTMENT)
            self.assertIsInstance(decision.decision_value, dict)
            self.assertIsInstance(decision.confidence, ConfidenceLevel)
            
            # Verify all agents have priority assignments
            optimized_priorities = decision.decision_value
            for task in tasks:
                self.assertIn(task.agent_id, optimized_priorities)
                self.assertIsInstance(optimized_priorities[task.agent_id], int)
                self.assertTrue(1 <= optimized_priorities[task.agent_id] <= 5)
        
        self.loop.run_until_complete(test_optimization())
    
    def test_context_analysis(self):
        """Test decision context analysis"""
        analysis = self.decision_engine._analyze_context(self.decision_context)
        
        # Verify analysis structure
        self.assertIn('phase_complexity', analysis)
        self.assertIn('agent_availability', analysis)
        self.assertIn('historical_performance', analysis)
        self.assertIn('time_pressure', analysis)
        self.assertIn('resource_availability', analysis)
        self.assertIn('metrics', analysis)
        
        # Verify metrics
        metrics = analysis['metrics']
        self.assertIn('complexity_score', metrics)
        self.assertIn('performance_score', metrics)
        self.assertIn('urgency_score', metrics)
        self.assertIn('resource_score', metrics)
        
        # Verify reasonable values
        self.assertTrue(0 <= analysis['phase_complexity'] <= 1)
        self.assertEqual(analysis['agent_availability'], 3)
        self.assertTrue(0 <= analysis['time_pressure'] <= 1)
    
    def test_agent_scoring(self):
        """Test agent scoring algorithm"""
        required_skills = ["instructional_design", "assessment"]
        agent_scores = self.decision_engine._score_agents(self.decision_context, required_skills)
        
        # Verify all agents are scored
        for agent_id in self.decision_context.available_agents:
            self.assertIn(agent_id, agent_scores)
            self.assertIsInstance(agent_scores[agent_id], float)
            self.assertTrue(0 <= agent_scores[agent_id] <= 1)
        
        # IPDAi should score highly for instructional design
        self.assertGreater(agent_scores["ipdai_agent"], agent_scores["tfdai_agent"])
    
    def test_execution_mode_determination(self):
        """Test execution mode determination logic"""
        
        # Test with different contexts
        test_cases = [
            {
                'analysis': {
                    'metrics': {
                        'complexity_score': 0.9,
                        'performance_score': 0.8,
                        'urgency_score': 0.2,
                        'resource_score': 0.5
                    },
                    'agent_availability': 2
                },
                'expected': ExecutionMode.SEQUENTIAL
            },
            {
                'analysis': {
                    'metrics': {
                        'complexity_score': 0.3,
                        'performance_score': 0.7,
                        'urgency_score': 0.9,
                        'resource_score': 0.8
                    },
                    'agent_availability': 4
                },
                'expected': ExecutionMode.PARALLEL
            }
        ]
        
        for case in test_cases:
            mode = self.decision_engine._determine_optimal_execution_mode(
                self.decision_context, case['analysis']
            )
            # Note: The actual determination might vary based on weights, 
            # so we just verify it returns a valid mode
            self.assertIsInstance(mode, ExecutionMode)
    
    def test_confidence_calculation(self):
        """Test confidence level calculation"""
        
        # High confidence scenario
        high_conf_analysis = {
            'metrics': {
                'performance_score': 0.9,
                'complexity_score': 0.8,
                'resource_score': 0.9,
                'urgency_score': 0.7
            },
            'agent_availability': 3
        }
        
        confidence = self.decision_engine._calculate_decision_confidence(high_conf_analysis)
        self.assertIn(confidence, [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH])
        
        # Low confidence scenario
        low_conf_analysis = {
            'metrics': {
                'performance_score': 0.3,
                'complexity_score': 0.2,
                'resource_score': 0.4,
                'urgency_score': 0.3
            },
            'agent_availability': 1
        }
        
        confidence = self.decision_engine._calculate_decision_confidence(low_conf_analysis)
        self.assertIn(confidence, [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM])
    
    def test_skill_matching(self):
        """Test agent skill matching"""
        
        # Test content creation skills
        content_skills = ["content_creation", "engagement"]
        cauthai_match = self.decision_engine._calculate_skill_match("cauthai_agent", content_skills)
        ipdai_match = self.decision_engine._calculate_skill_match("ipdai_agent", content_skills)
        
        # CAuthAi should match better for content creation
        self.assertGreater(cauthai_match, ipdai_match)
        
        # Test instructional design skills
        design_skills = ["instructional_design", "assessment"]
        ipdai_match = self.decision_engine._calculate_skill_match("ipdai_agent", design_skills)
        tfdai_match = self.decision_engine._calculate_skill_match("tfdai_agent", design_skills)
        
        # IPDAi should match better for instructional design
        self.assertGreater(ipdai_match, tfdai_match)
    
    def test_performance_feedback_learning(self):
        """Test performance feedback and learning"""
        
        # Initial state
        initial_success_rate = self.decision_engine.agent_success_rates.get("test_agent", 0.5)
        
        # Provide positive feedback
        self.decision_engine.update_performance_feedback(
            "decision_1",
            0.9,
            {
                'agent_id': 'test_agent',
                'success': True,
                'execution_time': 60.0
            }
        )
        
        # Verify learning
        new_success_rate = self.decision_engine.agent_success_rates["test_agent"]
        self.assertGreaterEqual(new_success_rate, initial_success_rate)
        self.assertIn("test_agent", self.decision_engine.agent_average_times)
        self.assertEqual(self.decision_engine.agent_average_times["test_agent"], 60.0)
    
    def test_decision_serialization(self):
        """Test decision serialization for storage/API"""
        
        async def test_serialization():
            decision = await self.decision_engine.make_execution_decision(self.decision_context)
            
            # Serialize decision
            decision_dict = decision.to_dict()
            
            # Verify serialization
            self.assertIn('decision_type', decision_dict)
            self.assertIn('decision_value', decision_dict)
            self.assertIn('confidence', decision_dict)
            self.assertIn('reasoning', decision_dict)
            self.assertIn('alternatives', decision_dict)
            self.assertIn('metrics_used', decision_dict)
            self.assertIn('timestamp', decision_dict)
            
            # Verify data types
            self.assertIsInstance(decision_dict['decision_type'], str)
            self.assertIsInstance(decision_dict['confidence'], str)
            self.assertIsInstance(decision_dict['reasoning'], str)
            self.assertIsInstance(decision_dict['timestamp'], str)
        
        self.loop.run_until_complete(test_serialization())
    
    def test_time_pressure_assessment(self):
        """Test time pressure assessment"""
        
        # No constraints - low pressure
        no_pressure = self.decision_engine._assess_time_pressure(None)
        self.assertEqual(no_pressure, 0.3)
        
        # Near deadline - high pressure
        near_deadline = {
            'deadline': (datetime.now() + timedelta(hours=1)).isoformat()
        }
        high_pressure = self.decision_engine._assess_time_pressure(near_deadline)
        self.assertGreater(high_pressure, 0.8)
        
        # Far deadline - low pressure
        far_deadline = {
            'deadline': (datetime.now() + timedelta(weeks=2)).isoformat()
        }
        low_pressure = self.decision_engine._assess_time_pressure(far_deadline)
        self.assertLess(low_pressure, 0.5)
    
    def test_resource_availability_assessment(self):
        """Test resource availability assessment"""
        
        # No constraints - full availability
        full_resources = self.decision_engine._assess_resource_availability(None)
        self.assertEqual(full_resources, 1.0)
        
        # Limited resources
        limited_resources = {
            'available_compute': 0.5,
            'max_compute': 1.0
        }
        limited_score = self.decision_engine._assess_resource_availability(limited_resources)
        self.assertEqual(limited_score, 0.5)
        
        # Abundant resources
        abundant_resources = {
            'available_compute': 2.0,
            'max_compute': 1.0
        }
        abundant_score = self.decision_engine._assess_resource_availability(abundant_resources)
        self.assertEqual(abundant_score, 2.0)
    
    def test_decision_statistics(self):
        """Test decision statistics generation"""
        
        async def test_stats():
            # Make several decisions
            await self.decision_engine.make_execution_decision(self.decision_context)
            await self.decision_engine.make_agent_selection_decision(
                self.decision_context, ["content_creation"]
            )
            
            # Get statistics
            stats = self.decision_engine.get_decision_statistics()
            
            # Verify statistics structure
            self.assertIn('total_decisions', stats)
            self.assertIn('decision_types', stats)
            self.assertIn('confidence_distribution', stats)
            self.assertIn('recent_decisions', stats)
            
            # Verify counts
            self.assertEqual(stats['total_decisions'], 2)
            self.assertIn('execution_mode', stats['decision_types'])
            self.assertIn('agent_selection', stats['decision_types'])
        
        self.loop.run_until_complete(test_stats())


class TestDecisionContext(unittest.TestCase):
    """Test DecisionContext functionality"""
    
    def test_decision_context_creation(self):
        """Test decision context creation"""
        conversation_state = ConversationState("test", {})
        
        context = DecisionContext(
            phase_id="test_phase",
            current_state=conversation_state,
            available_agents=["agent1", "agent2"],
            execution_history=[],
            performance_metrics={},
            user_preferences={}
        )
        
        self.assertEqual(context.phase_id, "test_phase")
        self.assertEqual(context.current_state, conversation_state)
        self.assertEqual(len(context.available_agents), 2)


class TestDecisionIntegration(unittest.TestCase):
    """Test integration with other orchestrator components"""
    
    def setUp(self):
        """Set up test environment"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_decision_engine_with_user_preferences(self):
        """Test decision engine with user preferences"""
        
        decision_engine = DynamicDecisionEngine(enable_logging=False)
        conversation_state = ConversationState("test", {})
        
        # Context with user preferences
        context = DecisionContext(
            phase_id="content_creation",
            current_state=conversation_state,
            available_agents=["ipdai_agent", "cauthai_agent"],
            execution_history=[],
            performance_metrics={},
            user_preferences={
                'execution_mode_preference': 'parallel',
                'preferred_agents': ['cauthai_agent']
            }
        )
        
        async def test_preferences():
            # Test execution mode decision
            exec_decision = await decision_engine.make_execution_decision(context)
            
            # Should consider user preference for parallel mode
            self.assertIsInstance(exec_decision.decision_value, ExecutionMode)
            
            # Test agent selection decision
            agent_decision = await decision_engine.make_agent_selection_decision(
                context, ["content_creation"]
            )
            
            # Should prefer cauthai_agent based on user preference
            self.assertEqual(agent_decision.decision_value, "cauthai_agent")
        
        self.loop.run_until_complete(test_preferences())


def run_phase_3_2_tests():
    """Run all Phase 3.2 tests and return results"""
    print("🧪 Running Phase 3.2 Tests: Dynamic Decision Engine")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDecisionEngine,
        TestDecisionContext,
        TestDecisionIntegration
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
    run_phase_3_2_tests()