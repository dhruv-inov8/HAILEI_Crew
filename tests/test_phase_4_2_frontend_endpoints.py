"""
Test Suite: Phase 4.2 - Frontend-Ready API Endpoints

Tests for frontend-specific endpoints, simplified interfaces,
and enhanced WebSocket functionality.
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

from fastapi.testclient import TestClient

# Import API components
from api.main import app
from api.auth import create_access_token
from api.frontend_endpoints import SimpleCourseRequest, ChatMessage
from api.frontend_websocket import frontend_ws_manager


class TestFrontendEndpoints(unittest.TestCase):
    """Test frontend-specific API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock the orchestrator for testing
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.active_sessions = {}
        self.mock_orchestrator.start_conversation = AsyncMock(return_value="frontend_test_session")
        
        # Create mock session state
        mock_session_state = {
            "current_phase": "foundation_design",
            "status": "active",
            "progress": 0.5,
            "agents": [],
            "context": {}
        }
        
        def mock_get_session_state(session_id):
            if session_id == "invalid_session":
                return None
            return mock_session_state
        
        def mock_load_session(session_id):
            if session_id == "invalid_session":
                return None
            mock_state = Mock()
            mock_state.current_phase = "foundation_design"
            return mock_state
        
        self.mock_orchestrator.get_session_state = Mock(side_effect=mock_get_session_state)
        self.mock_orchestrator.load_session = Mock(side_effect=mock_load_session)
        self.mock_orchestrator.store_global_context = Mock()
        self.mock_orchestrator.process_user_feedback = AsyncMock(return_value=Mock(to_dict=Mock(return_value={})))
        self.mock_orchestrator.approve_output = AsyncMock(return_value=True)
        self.mock_orchestrator.continue_workflow = AsyncMock()
        
        # Start patching - just patch main orchestrator as frontend imports it dynamically
        self.orchestrator_patcher = patch('api.main.orchestrator', self.mock_orchestrator)
        self.orchestrator_patcher.start()
        
        # Create test client
        self.client = TestClient(app)
        
        # Create test access token
        self.access_token = create_access_token(
            data={"sub": "frontend_user", "role": "user"}
        )
        self.headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
    
    def tearDown(self):
        """Clean up test environment"""
        self.orchestrator_patcher.stop()
    
    def test_quick_start_session(self):
        """Test frontend quick start endpoint"""
        course_data = {
            "title": "Frontend Course",
            "level": "Beginner",
            "duration": 8,
            "description": "A course for frontend development"
        }
        
        response = self.client.post(
            "/frontend/quick-start",
            json=course_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertIn("websocket_url", data)
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["session_id"], "frontend_test_session")
    
    def test_quick_start_with_preferences(self):
        """Test quick start with user preferences"""
        course_data = {
            "title": "Advanced Course",
            "level": "Advanced", 
            "duration": 12
        }
        
        # Note: preferences would be sent as query params or separate request in real implementation
        response = self.client.post(
            "/frontend/quick-start",
            json=course_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
    
    def test_chat_interface(self):
        """Test conversational chat interface"""
        # First create a session
        course_data = {
            "title": "Chat Test Course",
            "level": "Intermediate",
            "duration": 10
        }
        
        session_response = self.client.post(
            "/frontend/quick-start",
            json=course_data,
            headers=self.headers
        )
        session_id = session_response.json()["session_id"]
        
        # Test chat message
        chat_data = {
            "message": "Can you help me design the course structure?",
            "context": {"current_focus": "course_outline"}
        }
        
        response = self.client.post(
            f"/frontend/chat/{session_id}",
            json=chat_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("suggestions", data)
        self.assertIn("actions", data)
        self.assertIsInstance(data["suggestions"], list)
        self.assertIsInstance(data["actions"], list)
    
    def test_chat_invalid_session(self):
        """Test chat with invalid session"""
        chat_data = {
            "message": "Test message"
        }
        
        response = self.client.post(
            "/frontend/chat/invalid_session",
            json=chat_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 404)
    
    def test_workflow_progress(self):
        """Test workflow progress endpoint"""
        # Create session first
        course_data = {
            "title": "Progress Test",
            "level": "Beginner",
            "duration": 6
        }
        
        session_response = self.client.post(
            "/frontend/quick-start",
            json=course_data,
            headers=self.headers
        )
        session_id = session_response.json()["session_id"]
        
        # Get progress
        response = self.client.get(
            f"/frontend/progress/{session_id}",
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertIn("current_phase", data)
        self.assertIn("overall_progress", data)
        self.assertIn("agent_statuses", data)
        self.assertIsInstance(data["agent_statuses"], list)
        
        # Check agent status structure
        if data["agent_statuses"]:
            agent_status = data["agent_statuses"][0]
            self.assertIn("agent_id", agent_status)
            self.assertIn("status", agent_status)
            self.assertIn("progress", agent_status)
    
    def test_progress_invalid_session(self):
        """Test progress with invalid session"""
        response = self.client.get(
            "/frontend/progress/invalid_session",
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 404)
    
    def test_approve_action(self):
        """Test approval action endpoint"""
        # Create session
        course_data = {
            "title": "Approval Test",
            "level": "Intermediate",
            "duration": 8
        }
        
        session_response = self.client.post(
            "/frontend/quick-start",
            json=course_data,
            headers=self.headers
        )
        session_id = session_response.json()["session_id"]
        
        # Test approval
        response = self.client.post(
            f"/frontend/actions/{session_id}/approve",
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("success", data)
        self.assertIn("message", data)
        self.assertTrue(data["success"])
    
    def test_refinement_action(self):
        """Test refinement request endpoint"""
        # Create session
        course_data = {
            "title": "Refinement Test",
            "level": "Advanced",
            "duration": 12
        }
        
        session_response = self.client.post(
            "/frontend/quick-start",
            json=course_data,
            headers=self.headers
        )
        session_id = session_response.json()["session_id"]
        
        # Test refinement
        refinement_data = {
            "feedback": "Please add more interactive elements"
        }
        
        response = self.client.post(
            f"/frontend/actions/{session_id}/refine",
            json=refinement_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("success", data)
        self.assertIn("message", data)
        self.assertTrue(data["success"])
    
    def test_course_templates(self):
        """Test course templates endpoint"""
        response = self.client.get("/frontend/templates/course-types")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("templates", data)
        self.assertIsInstance(data["templates"], list)
        self.assertGreater(len(data["templates"]), 0)
        
        # Check template structure
        template = data["templates"][0]
        self.assertIn("id", template)
        self.assertIn("name", template)
        self.assertIn("description", template)
        self.assertIn("duration_weeks", template)
        self.assertIn("level", template)
    
    def test_frontend_health_check(self):
        """Test frontend-specific health check"""
        response = self.client.get("/frontend/health/frontend")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("services", data)
        self.assertIn("features", data)
        self.assertIn("orchestrator", data["services"])
        self.assertIn("chat_interface", data["features"])
    
    def test_unauthorized_frontend_access(self):
        """Test frontend endpoints without authentication"""
        # Templates should be accessible without auth
        response = self.client.get("/frontend/templates/course-types")
        self.assertEqual(response.status_code, 200)
        
        # Health check should be accessible
        response = self.client.get("/frontend/health/frontend")
        self.assertEqual(response.status_code, 200)
        
        # Quick start without auth should work (demo mode)
        course_data = {
            "title": "No Auth Test",
            "level": "Beginner",
            "duration": 4
        }
        
        response = self.client.post("/frontend/quick-start", json=course_data)
        self.assertEqual(response.status_code, 200)


class TestFrontendWebSocket(unittest.TestCase):
    """Test frontend WebSocket functionality"""
    
    def setUp(self):
        """Set up WebSocket test environment"""
        self.ws_manager = frontend_ws_manager
    
    def test_websocket_manager_initialization(self):
        """Test WebSocket manager initialization"""
        self.assertIsNotNone(self.ws_manager)
        self.assertIsNotNone(self.ws_manager.connection_manager)
        self.assertIsNotNone(self.ws_manager.event_handlers)
    
    def test_event_handlers_registration(self):
        """Test that all expected event handlers are registered"""
        expected_handlers = ["ping", "subscribe", "chat", "action", "status_request"]
        
        for handler in expected_handlers:
            self.assertIn(handler, self.ws_manager.event_handlers)
    
    def test_connection_stats(self):
        """Test connection statistics functionality"""
        stats = self.ws_manager.get_connection_stats()
        
        self.assertIn("total_connections", stats)
        self.assertIn("current_connections", stats)
        self.assertIsInstance(stats["current_connections"], int)


class TestFrontendModels(unittest.TestCase):
    """Test frontend-specific Pydantic models"""
    
    def test_simple_course_request_model(self):
        """Test SimpleCourseRequest model validation"""
        # Valid data
        valid_data = {
            "title": "Test Course",
            "level": "Beginner",
            "duration": 8,
            "description": "A test course"
        }
        
        course = SimpleCourseRequest(**valid_data)
        self.assertEqual(course.title, "Test Course")
        self.assertEqual(course.duration, 8)
        
        # Test without optional description
        minimal_data = {
            "title": "Minimal Course",
            "level": "Advanced",
            "duration": 12
        }
        
        course = SimpleCourseRequest(**minimal_data)
        self.assertEqual(course.title, "Minimal Course")
        self.assertIsNone(course.description)
    
    def test_chat_message_model(self):
        """Test ChatMessage model validation"""
        # Valid chat message
        chat_data = {
            "message": "Hello, can you help me?",
            "context": {"current_page": "course_design"}
        }
        
        chat = ChatMessage(**chat_data)
        self.assertEqual(chat.message, "Hello, can you help me?")
        self.assertIn("current_page", chat.context)
        
        # Message without context
        simple_chat = ChatMessage(message="Simple message")
        self.assertEqual(simple_chat.message, "Simple message")
        self.assertIsNone(simple_chat.context)


def run_phase_4_2_tests():
    """Run all Phase 4.2 tests and return results"""
    print("🧪 Running Phase 4.2 Tests: Frontend-Ready API Endpoints")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFrontendEndpoints,
        TestFrontendWebSocket,
        TestFrontendModels
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    run_phase_4_2_tests()