"""
Test Suite: Phase 3.4 - Comprehensive Error Recovery System

Tests for error detection, classification, recovery strategies,
circuit breaker patterns, and recovery learning mechanisms.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.error_recovery import (
    ErrorRecoverySystem,
    ErrorContext,
    RecoveryAction,
    RecoveryResult,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy
)
from orchestrator.conversation_state import ConversationState, AgentOutput


class TestErrorContext(unittest.TestCase):
    """Test ErrorContext functionality"""
    
    def test_error_context_creation(self):
        """Test error context creation and serialization"""
        error_context = ErrorContext(
            error_id="test_error_001",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT_EXECUTION,
            component="ipdai_agent",
            operation="create_foundation",
            error_message="Agent execution failed",
            session_id="test_session",
            phase_id="foundation_design",
            agent_id="ipdai_agent"
        )
        
        self.assertEqual(error_context.error_id, "test_error_001")
        self.assertEqual(error_context.severity, ErrorSeverity.HIGH)
        self.assertEqual(error_context.category, ErrorCategory.AGENT_EXECUTION)
        self.assertEqual(error_context.agent_id, "ipdai_agent")
        
        # Test serialization
        serialized = error_context.to_dict()
        self.assertIn('error_id', serialized)
        self.assertIn('severity', serialized)
        self.assertIn('category', serialized)
        self.assertEqual(serialized['severity'], 'high')
        self.assertEqual(serialized['category'], 'agent_execution')


class TestRecoveryAction(unittest.TestCase):
    """Test RecoveryAction functionality"""
    
    def test_recovery_action_creation(self):
        """Test recovery action creation"""
        action = RecoveryAction(
            action_id="retry_001",
            strategy=RecoveryStrategy.RETRY,
            description="Retry agent execution",
            parameters={"max_retries": 3},
            max_attempts=3,
            timeout_seconds=60
        )
        
        self.assertEqual(action.strategy, RecoveryStrategy.RETRY)
        self.assertEqual(action.max_attempts, 3)
        self.assertEqual(action.parameters["max_retries"], 3)
        
        # Test serialization
        serialized = action.to_dict()
        self.assertEqual(serialized['strategy'], 'retry')
        self.assertEqual(serialized['max_attempts'], 3)


class TestErrorRecoverySystem(unittest.TestCase):
    """Test Error Recovery System functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock orchestrator
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.active_sessions = {}
        self.mock_orchestrator.conversation_state = ConversationState(
            session_id="test_session",
            course_request={"course_title": "Test Course"}
        )
        self.mock_orchestrator.conversation_state.set_current_phase("content_creation")
        
        # Create error recovery system
        self.error_recovery = ErrorRecoverySystem(
            orchestrator=self.mock_orchestrator,
            enable_logging=False,
            max_error_history=100
        )
        
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test environment"""
        self.loop.close()
    
    def test_error_recovery_initialization(self):
        """Test error recovery system initialization"""
        self.assertIsNotNone(self.error_recovery.error_history)
        self.assertIsNotNone(self.error_recovery.recovery_strategies)
        self.assertIsNotNone(self.error_recovery.circuit_breakers)
        self.assertEqual(len(self.error_recovery.error_history), 0)
    
    def test_error_severity_classification(self):
        """Test error severity classification"""
        
        # Test critical errors
        critical_error = SystemExit("System shutdown")
        severity = self.error_recovery._classify_error_severity(critical_error, {})
        self.assertEqual(severity, ErrorSeverity.CRITICAL)
        
        # Test high severity errors
        high_error = ConnectionError("Connection failed")
        severity = self.error_recovery._classify_error_severity(high_error, {})
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        # Test phase-based severity
        phase_error = ValueError("Invalid input")
        context = {"phase_id": "final_integration"}
        severity = self.error_recovery._classify_error_severity(phase_error, context)
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        # Test default severity
        generic_error = RuntimeError("Generic error")
        severity = self.error_recovery._classify_error_severity(generic_error, {})
        self.assertEqual(severity, ErrorSeverity.MEDIUM)
    
    def test_error_category_classification(self):
        """Test error category classification"""
        
        # Test component-based classification
        category = self.error_recovery._classify_error_category(
            ValueError("Test"), "agent_ipdai", "execute"
        )
        self.assertEqual(category, ErrorCategory.AGENT_EXECUTION)
        
        category = self.error_recovery._classify_error_category(
            ValueError("Test"), "context_manager", "store"
        )
        self.assertEqual(category, ErrorCategory.CONTEXT_MANAGEMENT)
        
        # Test error type-based classification
        category = self.error_recovery._classify_error_category(
            ConnectionError("Test"), "unknown", "unknown"
        )
        self.assertEqual(category, ErrorCategory.COMMUNICATION)
        
        category = self.error_recovery._classify_error_category(
            ValueError("Test"), "unknown", "unknown"
        )
        self.assertEqual(category, ErrorCategory.DATA_VALIDATION)
    
    def test_basic_error_handling(self):
        """Test basic error handling workflow"""
        
        async def test_error_handling():
            # Create test error
            test_error = ValueError("Test error for recovery")
            context = {
                "session_id": "test_session",
                "phase_id": "content_creation",
                "agent_id": "test_agent"
            }
            
            # Handle error
            recovery_result = await self.error_recovery.handle_error(
                error=test_error,
                context=context,
                operation="test_operation",
                component="test_component"
            )
            
            # Verify recovery result structure
            self.assertIsInstance(recovery_result, RecoveryResult)
            self.assertIsInstance(recovery_result.action_taken, RecoveryAction)
            self.assertIsInstance(recovery_result.success, bool)
            self.assertGreaterEqual(recovery_result.attempts_made, 1)
            self.assertGreaterEqual(recovery_result.time_taken, 0)
            
            # Verify error was recorded
            self.assertEqual(len(self.error_recovery.error_history), 1)
            recorded_error = self.error_recovery.error_history[0]
            self.assertEqual(recorded_error.error_message, "Test error for recovery")
            self.assertEqual(recorded_error.session_id, "test_session")
        
        self.loop.run_until_complete(test_error_handling())
    
    def test_agent_failure_recovery(self):
        """Test specific agent failure recovery"""
        
        async def test_agent_recovery():
            test_error = RuntimeError("Agent execution failed")
            
            result = await self.error_recovery.recover_from_agent_failure(
                agent_id="ipdai_agent",
                task_description="Create course foundation",
                error=test_error,
                context={"phase_id": "foundation_design"}
            )
            
            # Should return an AgentOutput or None
            self.assertTrue(result is None or isinstance(result, AgentOutput))
            
            # Verify error was recorded with agent context
            self.assertEqual(len(self.error_recovery.error_history), 1)
            recorded_error = self.error_recovery.error_history[0]
            self.assertEqual(recorded_error.category, ErrorCategory.AGENT_EXECUTION)
            self.assertEqual(recorded_error.component, "agent_ipdai_agent")
        
        self.loop.run_until_complete(test_agent_recovery())
    
    def test_parallel_execution_failure_recovery(self):
        """Test parallel execution failure recovery"""
        
        async def test_parallel_recovery():
            test_error = TimeoutError("Parallel execution timeout")
            failed_agents = ["agent1", "agent2"]
            
            result = await self.error_recovery.recover_from_parallel_execution_failure(
                phase_id="content_creation",
                failed_agents=failed_agents,
                error=test_error,
                context={"execution_mode": "parallel"}
            )
            
            # Verify recovery context structure
            self.assertIn('recovery_success', result)
            self.assertIn('recovery_action', result)
            self.assertIn('failed_agents', result)
            self.assertIn('phase_id', result)
            self.assertEqual(result['failed_agents'], failed_agents)
            self.assertEqual(result['phase_id'], "content_creation")
        
        self.loop.run_until_complete(test_parallel_recovery())
    
    def test_retry_strategy_execution(self):
        """Test retry strategy execution"""
        
        async def test_retry():
            error_context = ErrorContext(
                error_id="retry_test",
                timestamp=datetime.now(),
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.AGENT_EXECUTION,
                component="test_agent",
                operation="test_operation",
                error_message="Retriable error",
                agent_id="test_agent"
            )
            
            recovery_action = RecoveryAction(
                action_id="retry_action",
                strategy=RecoveryStrategy.RETRY,
                description="Test retry",
                max_attempts=2
            )
            
            result = await self.error_recovery._execute_strategy(
                error_context, recovery_action, 0
            )
            
            # Should return some result indicating retry attempt
            self.assertIsNotNone(result)
            # Could be either dict (for generic retry) or AgentOutput (for agent retry)
            self.assertTrue(isinstance(result, (dict, AgentOutput)))
        
        self.loop.run_until_complete(test_retry())
    
    def test_fallback_strategy_execution(self):
        """Test fallback strategy execution"""
        
        async def test_fallback():
            error_context = ErrorContext(
                error_id="fallback_test",
                timestamp=datetime.now(),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.AGENT_EXECUTION,
                component="test_agent",
                operation="test_operation",
                error_message="Fallback needed",
                agent_id="test_agent"
            )
            
            recovery_action = RecoveryAction(
                action_id="fallback_action",
                strategy=RecoveryStrategy.FALLBACK,
                description="Test fallback"
            )
            
            result = await self.error_recovery._execute_strategy(
                error_context, recovery_action, 0
            )
            
            # Should return fallback result
            self.assertIsNotNone(result)
            if isinstance(result, dict):
                self.assertIn('fallback_applied', result)
        
        self.loop.run_until_complete(test_fallback())
    
    def test_graceful_degradation_strategy(self):
        """Test graceful degradation strategy"""
        
        async def test_degradation():
            error_context = ErrorContext(
                error_id="degradation_test",
                timestamp=datetime.now(),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM_RESOURCE,
                component="system",
                operation="resource_allocation",
                error_message="Resource exhausted"
            )
            
            recovery_action = RecoveryAction(
                action_id="degradation_action",
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                description="Test degradation"
            )
            
            result = await self.error_recovery._execute_strategy(
                error_context, recovery_action, 0
            )
            
            # Should return degraded mode result
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            self.assertIn('degraded_mode', result)
            self.assertTrue(result['degraded_mode'])
        
        self.loop.run_until_complete(test_degradation())
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern"""
        
        # Test circuit breaker starts closed
        self.assertFalse(
            self.error_recovery._is_circuit_breaker_open("test_component", "test_operation")
        )
        
        # Simulate multiple failures to open circuit breaker
        for _ in range(6):  # Threshold is 5 failures
            self.error_recovery._update_circuit_breaker("test_component", "test_operation", False)
        
        # Circuit breaker should now be open
        self.assertTrue(
            self.error_recovery._is_circuit_breaker_open("test_component", "test_operation")
        )
        
        # Test success resets failure count
        self.error_recovery._update_circuit_breaker("test_component", "test_operation2", False)
        self.error_recovery._update_circuit_breaker("test_component", "test_operation2", True)
        self.assertFalse(
            self.error_recovery._is_circuit_breaker_open("test_component", "test_operation2")
        )
    
    def test_custom_error_handler_registration(self):
        """Test custom error handler registration"""
        
        def custom_handler(error, context):
            return RecoveryAction(
                action_id="custom_recovery",
                strategy=RecoveryStrategy.ALTERNATIVE_PATH,
                description="Custom recovery strategy"
            )
        
        # Register custom handler
        self.error_recovery.register_custom_handler(ValueError, custom_handler)
        
        # Verify handler is registered
        self.assertIn(ValueError, self.error_recovery.custom_handlers)
        self.assertEqual(self.error_recovery.custom_handlers[ValueError], custom_handler)
    
    def test_error_pattern_tracking(self):
        """Test error pattern analysis"""
        
        async def test_patterns():
            # Create multiple similar errors
            for i in range(3):
                await self.error_recovery.handle_error(
                    error=ValueError(f"Pattern error {i}"),
                    context={"test": True},
                    operation="pattern_test",
                    component="pattern_component"
                )
            
            # Check pattern tracking
            pattern_key = "data_validation_pattern_component_pattern_test"
            self.assertIn(pattern_key, self.error_recovery.error_patterns)
            self.assertEqual(self.error_recovery.error_patterns[pattern_key], 3)
        
        self.loop.run_until_complete(test_patterns())
    
    def test_recovery_learning(self):
        """Test recovery learning mechanism"""
        
        # Create test error context and recovery action
        error_context = ErrorContext(
            error_id="learning_test",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AGENT_EXECUTION,
            component="test_agent",
            operation="test_operation",
            error_message="Learning test error"
        )
        
        recovery_action = RecoveryAction(
            action_id="learning_action",
            strategy=RecoveryStrategy.RETRY,
            description="Learning test action"
        )
        
        recovery_result = RecoveryResult(
            success=True,
            action_taken=recovery_action,
            attempts_made=2,
            time_taken=1.5
        )
        
        # Test learning
        self.error_recovery._learn_from_recovery(error_context, recovery_action, recovery_result)
        
        # Verify learning data was stored
        strategy_key = "agent_execution_retry"
        self.assertIn(strategy_key, self.error_recovery.strategy_effectiveness)
        self.assertEqual(len(self.error_recovery.strategy_effectiveness[strategy_key]), 1)
        self.assertEqual(self.error_recovery.strategy_effectiveness[strategy_key][0], 1.0)
        
        # Test unsuccessful recovery learning
        failed_result = RecoveryResult(
            success=False,
            action_taken=recovery_action,
            attempts_made=3,
            time_taken=2.0,
            error_message="Recovery failed"
        )
        
        self.error_recovery._learn_from_recovery(error_context, recovery_action, failed_result)
        self.assertEqual(len(self.error_recovery.strategy_effectiveness[strategy_key]), 2)
        self.assertEqual(self.error_recovery.strategy_effectiveness[strategy_key][1], 0.0)
    
    def test_error_statistics_generation(self):
        """Test error statistics generation"""
        
        async def test_stats():
            # Generate some test errors
            await self.error_recovery.handle_error(
                ValueError("Test error 1"), {"test": True}, "op1", "comp1"
            )
            await self.error_recovery.handle_error(
                ConnectionError("Test error 2"), {"test": True}, "op2", "comp2"
            )
            await self.error_recovery.handle_error(
                RuntimeError("Test error 3"), {"test": True}, "op3", "comp1"
            )
            
            # Get statistics
            stats = self.error_recovery.get_error_statistics()
            
            # Verify statistics structure
            self.assertIn('total_errors', stats)
            self.assertIn('category_distribution', stats)
            self.assertIn('severity_distribution', stats)
            self.assertIn('component_distribution', stats)
            
            # Verify counts
            self.assertEqual(stats['total_errors'], 3)
            self.assertIn('data_validation', stats['category_distribution'])
            self.assertIn('communication', stats['category_distribution'])
            self.assertEqual(stats['component_distribution']['comp1'], 2)
        
        self.loop.run_until_complete(test_stats())
    
    def test_user_intervention_strategy(self):
        """Test user intervention strategy"""
        
        async def test_user_intervention():
            error_context = ErrorContext(
                error_id="intervention_test",
                timestamp=datetime.now(),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.USER_INTERACTION,
                component="user_interface",
                operation="user_input",
                error_message="User intervention required"
            )
            
            recovery_action = RecoveryAction(
                action_id="intervention_action",
                strategy=RecoveryStrategy.USER_INTERVENTION,
                description="Request user intervention"
            )
            
            result = await self.error_recovery._execute_strategy(
                error_context, recovery_action, 0
            )
            
            # Should return intervention request
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            self.assertIn('user_intervention_required', result)
            self.assertTrue(result['user_intervention_required'])
            self.assertIn('suggested_actions', result)
        
        self.loop.run_until_complete(test_user_intervention())
    
    def test_skip_and_continue_strategy(self):
        """Test skip and continue strategy"""
        
        async def test_skip():
            error_context = ErrorContext(
                error_id="skip_test",
                timestamp=datetime.now(),
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.AGENT_EXECUTION,
                component="optional_agent",
                operation="optional_task",
                error_message="Optional operation failed"
            )
            
            recovery_action = RecoveryAction(
                action_id="skip_action",
                strategy=RecoveryStrategy.SKIP_AND_CONTINUE,
                description="Skip failed operation"
            )
            
            result = await self.error_recovery._execute_strategy(
                error_context, recovery_action, 0
            )
            
            # Should return skip confirmation
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            self.assertIn('operation_skipped', result)
            self.assertTrue(result['operation_skipped'])
            self.assertIn('continuation', result)
        
        self.loop.run_until_complete(test_skip())


def run_phase_3_4_tests():
    """Run all Phase 3.4 tests and return results"""
    print("🧪 Running Phase 3.4 Tests: Comprehensive Error Recovery System")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestErrorContext,
        TestRecoveryAction,
        TestErrorRecoverySystem
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
    run_phase_3_4_tests()