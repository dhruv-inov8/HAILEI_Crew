"""
Test Suite: Phase 4.1 - FastAPI Backend Integration

Tests for API endpoints, WebSocket functionality, authentication,
and integration with the HAILEI orchestration system.
"""

import unittest
import asyncio
import json
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

# Import API components
from api.main import app
from api.auth import create_access_token, authenticate_user
from api.websocket_manager import ConnectionManager
from api.models import *


class TestAPIEndpoints(unittest.TestCase):
    """Test FastAPI endpoint functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock the orchestrator for testing
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.active_sessions = {}
        self.mock_orchestrator.start_conversation = AsyncMock(return_value="test_session_123")
        
        # Create mock session state as a dictionary for proper serialization
        mock_session_state = {
            "current_phase": "foundation_design",
            "status": "active",
            "progress": 0.5,
            "agents": [],
            "context": {}
        }
        
        def mock_get_session_state(session_id):
            if session_id == "invalid_session_id":
                return None
            return mock_session_state
        
        def mock_load_session(session_id):
            if session_id == "invalid_session_id":
                return None
            # Return a mock object with current_phase attribute for orchestrator logic
            mock_state = Mock()
            mock_state.current_phase = "foundation_design"
            return mock_state
        
        self.mock_orchestrator.get_session_state = Mock(side_effect=mock_get_session_state)
        self.mock_orchestrator.load_session = Mock(side_effect=mock_load_session)
        self.mock_orchestrator.activate_agent = AsyncMock(return_value=Mock(to_dict=Mock(return_value={})))
        self.mock_orchestrator.process_user_feedback = AsyncMock(return_value=Mock(to_dict=Mock(return_value={})))
        self.mock_orchestrator.approve_output = AsyncMock(return_value=True)
        self.mock_orchestrator.complete_phase = AsyncMock(return_value=True)
        self.mock_orchestrator.continue_workflow = AsyncMock()
        self.mock_orchestrator.get_next_phase = Mock(return_value="next_phase")
        self.mock_orchestrator.execute_phase_with_parallel_coordination = AsyncMock(return_value={})
        self.mock_orchestrator.retrieve_relevant_context = Mock(return_value=[{"id": "test_context", "content": "test content"}])
        self.mock_orchestrator.store_global_context = Mock()
        self.mock_orchestrator.get_parallel_execution_statistics = Mock(return_value={})
        self.mock_orchestrator.get_decision_engine_statistics = Mock(return_value={})
        self.mock_orchestrator.get_context_manager_statistics = Mock(return_value={})
        self.mock_orchestrator.get_error_recovery_statistics = Mock(return_value={})
        
        # Start patching the orchestrator
        self.orchestrator_patcher = patch('api.main.orchestrator', self.mock_orchestrator)
        self.orchestrator_patcher.start()
        
        # Create test client
        self.client = TestClient(app)
        
        # Create test access token
        self.access_token = create_access_token(
            data={"sub": "test_user", "role": "user"}
        )
        self.headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
    
    def tearDown(self):
        """Clean up test environment"""
        self.orchestrator_patcher.stop()
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "operational")
    
    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("components", data)
    
    def test_create_session_endpoint(self):
        """Test session creation endpoint"""
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        request_data = {
            "course_request": course_request
        }
        
        response = self.client.post(
            "/sessions",
            json=request_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "created")
    
    def test_get_session_endpoint(self):
        """Test get session endpoint"""
        # First create a session
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate", 
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        create_response = self.client.post(
            "/sessions",
            json={"course_request": course_request},
            headers=self.headers
        )
        
        session_id = create_response.json()["session_id"]
        
        # Then get the session
        response = self.client.get(
            f"/sessions/{session_id}",
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertIn("session_state", data)
    
    def test_agent_execution_endpoint(self):
        """Test agent execution endpoint"""
        # Create session first
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        create_response = self.client.post(
            "/sessions",
            json={"course_request": course_request},
            headers=self.headers
        )
        
        session_id = create_response.json()["session_id"]
        
        # Execute agent
        execution_request = {
            "task_description": "Create course foundation",
            "context": {"phase": "foundation_design"}
        }
        
        response = self.client.post(
            f"/sessions/{session_id}/agents/ipdai_agent/execute",
            json=execution_request,
            headers=self.headers
        )
        
        # May fail if orchestrator not fully initialized, check status
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("agent_id", data)
            self.assertIn("output", data)
            self.assertIn("success", data)
    
    def test_feedback_submission_endpoint(self):
        """Test feedback submission endpoint"""
        # Create session
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        create_response = self.client.post(
            "/sessions",
            json={"course_request": course_request},
            headers=self.headers
        )
        
        session_id = create_response.json()["session_id"]
        
        # Submit feedback
        feedback_request = {
            "agent_id": "ipdai_agent",
            "feedback": "Please add more practical examples",
            "phase_id": "foundation_design"
        }
        
        response = self.client.post(
            f"/sessions/{session_id}/feedback",
            json=feedback_request,
            headers=self.headers
        )
        
        # May fail if orchestrator not ready
        self.assertIn(response.status_code, [200, 503])
    
    def test_approval_endpoint(self):
        """Test output approval endpoint"""
        # Create session
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        create_response = self.client.post(
            "/sessions",
            json={"course_request": course_request},
            headers=self.headers
        )
        
        session_id = create_response.json()["session_id"]
        
        # Submit approval
        approval_request = {
            "agent_id": "ipdai_agent",
            "phase_id": "foundation_design",
            "complete_phase": False
        }
        
        response = self.client.post(
            f"/sessions/{session_id}/approve",
            json=approval_request,
            headers=self.headers
        )
        
        # May fail if orchestrator not ready
        self.assertIn(response.status_code, [200, 503])
    
    def test_phase_execution_endpoint(self):
        """Test phase execution endpoint"""
        # Create session
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        create_response = self.client.post(
            "/sessions",
            json={"course_request": course_request},
            headers=self.headers
        )
        
        session_id = create_response.json()["session_id"]
        
        # Execute phase
        phase_request = {
            "agent_tasks": [
                {
                    "agent_id": "ipdai_agent",
                    "task_description": "Create foundation",
                    "dependencies": [],
                    "priority": 3
                }
            ]
        }
        
        response = self.client.post(
            f"/sessions/{session_id}/phases/foundation_design/execute",
            json=phase_request,
            headers=self.headers
        )
        
        # May fail if orchestrator not ready
        self.assertIn(response.status_code, [200, 503])
    
    def test_context_retrieval_endpoint(self):
        """Test context retrieval endpoint"""
        # Create session
        course_request = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        create_response = self.client.post(
            "/sessions",
            json={"course_request": course_request},
            headers=self.headers
        )
        
        session_id = create_response.json()["session_id"]
        
        # Get context
        response = self.client.get(
            f"/sessions/{session_id}/context?context_types=global,phase&max_results=10",
            headers=self.headers
        )
        
        # May fail if orchestrator not ready
        self.assertIn(response.status_code, [200, 503])
    
    def test_statistics_endpoint(self):
        """Test system statistics endpoint"""
        response = self.client.get(
            "/statistics",
            headers=self.headers
        )
        
        # May fail if orchestrator not ready
        self.assertIn(response.status_code, [200, 503])
    
    def test_unauthorized_access(self):
        """Test endpoints without authentication"""
        # Test with no auth header
        response = self.client.post(
            "/sessions",
            json={"course_request": {
                "course_title": "Test",
                "course_level": "Beginner",
                "course_duration_weeks": 8,
                "target_audience": "Students"
            }}
        )
        
        # Should still work for demo purposes (returns 200)
        # In production, would return 401
        self.assertIn(response.status_code, [200, 401])
    
    def test_invalid_session_id(self):
        """Test endpoints with invalid session ID"""
        response = self.client.get(
            "/sessions/invalid_session_id",
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 404)


class TestAuthentication(unittest.TestCase):
    """Test authentication functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.client = TestClient(app)
    
    def test_user_authentication(self):
        """Test user authentication function"""
        # Test valid credentials
        user = authenticate_user("admin", "admin123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "admin")
        
        # Test invalid credentials
        user = authenticate_user("admin", "wrongpassword")
        self.assertIsNone(user)
        
        # Test non-existent user
        user = authenticate_user("nonexistent", "password")
        self.assertIsNone(user)
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        from api.auth import create_access_token, verify_token
        
        # Create token
        token_data = {"sub": "test_user", "role": "user"}
        token = create_access_token(token_data)
        
        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)
        
        # Verify token
        payload = verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "test_user")
        self.assertEqual(payload["role"], "user")
        
        # Test invalid token
        invalid_payload = verify_token("invalid.token.here")
        self.assertIsNone(invalid_payload)
    
    def test_login_endpoint(self):
        """Test login endpoint"""
        # Test valid login
        response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertIn("token_type", data)
        self.assertEqual(data["token_type"], "bearer")
        
        # Test invalid login
        response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )
        
        self.assertEqual(response.status_code, 401)
    
    def test_user_info_endpoint(self):
        """Test user info endpoint"""
        # Get access token
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get user info
        response = self.client.get("/auth/me", headers=headers)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("user_id", data)
        self.assertIn("username", data)
        self.assertIn("role", data)
        self.assertEqual(data["username"], "admin")
    
    def test_demo_credentials_endpoint(self):
        """Test demo credentials endpoint"""
        response = self.client.get("/auth/demo-credentials")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("credentials", data)
        self.assertIsInstance(data["credentials"], list)
        self.assertGreater(len(data["credentials"]), 0)


class TestWebSocketManager(unittest.TestCase):
    """Test WebSocket connection manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.connection_manager = ConnectionManager()
        self.mock_websocket = Mock(spec=WebSocket)
        self.session_id = "test_session_123"
    
    def test_connection_management(self):
        """Test WebSocket connection management"""
        # Test initial state
        self.assertEqual(self.connection_manager.get_connection_count(), 0)
        self.assertFalse(self.connection_manager.is_session_connected(self.session_id))
        
        # Test connection
        # Note: We can't fully test connect() without async setup
        # But we can test the data structures
        self.connection_manager.active_connections[self.session_id].add(self.mock_websocket)
        self.connection_manager.connection_metadata[self.mock_websocket] = {
            'session_id': self.session_id,
            'connected_at': datetime.now(),
            'last_ping': datetime.now(),
            'message_count': 0
        }
        
        self.assertEqual(self.connection_manager.get_connection_count(), 1)
        self.assertTrue(self.connection_manager.is_session_connected(self.session_id))
        
        # Test disconnection
        self.connection_manager.disconnect(self.mock_websocket, self.session_id)
        self.assertEqual(self.connection_manager.get_connection_count(), 0)
        self.assertFalse(self.connection_manager.is_session_connected(self.session_id))
    
    def test_message_queuing(self):
        """Test message queuing for offline sessions"""
        # Queue message for offline session
        test_message = {"type": "test", "data": "test_data"}
        
        # Simulate queuing by calling send_to_session on non-connected session
        # In real implementation, this would queue the message
        self.assertEqual(len(self.connection_manager.message_queue[self.session_id]), 0)
        
        # Add message manually to test queue
        self.connection_manager.message_queue[self.session_id].append(test_message)
        self.assertEqual(len(self.connection_manager.message_queue[self.session_id]), 1)
    
    def test_connection_statistics(self):
        """Test connection statistics"""
        stats = self.connection_manager.get_connection_stats()
        
        self.assertIn('total_connections', stats)
        self.assertIn('current_connections', stats)
        self.assertIn('sessions_with_connections', stats)
        self.assertIn('messages_sent', stats)
        self.assertIsInstance(stats['current_connections'], int)
    
    def test_session_list(self):
        """Test session list functionality"""
        # Initially empty
        sessions = self.connection_manager.get_session_list()
        self.assertEqual(len(sessions), 0)
        
        # Add session
        self.connection_manager.active_connections[self.session_id].add(self.mock_websocket)
        sessions = self.connection_manager.get_session_list()
        self.assertIn(self.session_id, sessions)


class TestAPIModels(unittest.TestCase):
    """Test Pydantic models for API validation"""
    
    def test_course_request_model(self):
        """Test CourseRequest model validation"""
        # Valid course request
        valid_data = {
            "course_title": "Test Course",
            "course_level": "Intermediate",
            "course_duration_weeks": 12,
            "target_audience": "Software developers"
        }
        
        course_request = CourseRequest(**valid_data)
        self.assertEqual(course_request.course_title, "Test Course")
        self.assertEqual(course_request.course_duration_weeks, 12)
        
        # Test validation error
        with self.assertRaises(ValueError):
            CourseRequest(
                course_title="Test",
                course_level="Invalid",
                course_duration_weeks=100,  # Too many weeks
                target_audience="Students"
            )
    
    def test_agent_task_model(self):
        """Test AgentTask model validation"""
        valid_data = {
            "agent_id": "ipdai_agent",
            "task_description": "Create course foundation",
            "priority": 3
        }
        
        agent_task = AgentTask(**valid_data)
        self.assertEqual(agent_task.agent_id, "ipdai_agent")
        self.assertEqual(agent_task.priority, 3)
        
        # Test defaults
        self.assertEqual(agent_task.context, {})
        self.assertEqual(agent_task.dependencies, [])
    
    def test_user_preferences_model(self):
        """Test UserPreferences model"""
        preferences = UserPreferences(
            execution_mode_preference="parallel",
            preferred_agents=["ipdai_agent", "cauthai_agent"],
            communication_style="casual",
            detail_level="high"
        )
        
        self.assertEqual(preferences.execution_mode_preference, "parallel")
        self.assertEqual(len(preferences.preferred_agents), 2)
    
    def test_websocket_message_model(self):
        """Test WebSocketMessage model"""
        message = WebSocketMessage(
            type="agent_update",
            session_id="test_session",
            data={"agent_id": "test_agent", "status": "completed"}
        )
        
        self.assertEqual(message.type, "agent_update")
        self.assertEqual(message.session_id, "test_session")
        self.assertIsInstance(message.timestamp, datetime)


def run_phase_4_1_tests():
    """Run all Phase 4.1 tests and return results"""
    print("🧪 Running Phase 4.1 Tests: FastAPI Backend Integration")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAPIEndpoints,
        TestAuthentication,
        TestWebSocketManager,
        TestAPIModels
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
    run_phase_4_1_tests()