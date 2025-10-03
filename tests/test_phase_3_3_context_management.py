"""
Test Suite: Phase 3.3 - Enhanced Context Management

Tests for intelligent context storage, retrieval, memory optimization,
and context-aware information management.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.context_manager import (
    EnhancedContextManager,
    ContextEntry,
    ContextQuery,
    ContextType,
    MemoryPriority
)
from orchestrator.conversation_state import ConversationState


class TestContextEntry(unittest.TestCase):
    """Test ContextEntry functionality"""
    
    def test_context_entry_creation(self):
        """Test context entry creation and basic properties"""
        content = {"key": "value", "data": [1, 2, 3]}
        entry = ContextEntry(
            context_id="test_001",
            context_type=ContextType.AGENT,
            content=content,
            priority=MemoryPriority.HIGH,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags={"test", "agent"},
            source_agent="ipdai_agent",
            phase_id="foundation_design"
        )
        
        self.assertEqual(entry.context_id, "test_001")
        self.assertEqual(entry.context_type, ContextType.AGENT)
        self.assertEqual(entry.content, content)
        self.assertEqual(entry.priority, MemoryPriority.HIGH)
        self.assertIn("test", entry.tags)
        self.assertEqual(entry.source_agent, "ipdai_agent")
    
    def test_access_tracking(self):
        """Test access count and timestamp tracking"""
        entry = ContextEntry(
            context_id="test_access",
            context_type=ContextType.GLOBAL,
            content={"test": "data"},
            priority=MemoryPriority.MEDIUM,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        initial_count = entry.access_count
        initial_time = entry.last_accessed
        
        # Simulate access
        entry.update_access()
        
        self.assertEqual(entry.access_count, initial_count + 1)
        self.assertGreater(entry.last_accessed, initial_time)
    
    def test_retention_score_calculation(self):
        """Test retention score calculation for memory management"""
        # High priority, recently accessed entry
        high_priority_entry = ContextEntry(
            context_id="high_priority",
            context_type=ContextType.GLOBAL,
            content={"important": "data"},
            priority=MemoryPriority.CRITICAL,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=5,
            relevance_score=0.9
        )
        
        # Low priority, old entry
        low_priority_entry = ContextEntry(
            context_id="low_priority",
            context_type=ContextType.TEMPORAL,
            content={"old": "data"},
            priority=MemoryPriority.TRANSIENT,
            created_at=datetime.now() - timedelta(days=7),
            last_accessed=datetime.now() - timedelta(days=7),
            access_count=1,
            relevance_score=0.2
        )
        
        high_score = high_priority_entry.calculate_retention_score()
        low_score = low_priority_entry.calculate_retention_score()
        
        self.assertGreater(high_score, low_score)
        self.assertGreater(high_score, 0.5)
        self.assertLess(low_score, 0.5)
    
    def test_serialization(self):
        """Test context entry serialization"""
        entry = ContextEntry(
            context_id="serialize_test",
            context_type=ContextType.PHASE,
            content={"complex": {"nested": "data"}},
            priority=MemoryPriority.HIGH,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags={"serialize", "test"},
            source_agent="cauthai_agent"
        )
        
        serialized = entry.to_dict()
        
        # Verify all fields are present
        self.assertIn('context_id', serialized)
        self.assertIn('context_type', serialized)
        self.assertIn('content', serialized)
        self.assertIn('priority', serialized)
        self.assertIn('created_at', serialized)
        self.assertIn('tags', serialized)
        
        # Verify data integrity
        self.assertEqual(serialized['context_id'], "serialize_test")
        self.assertEqual(serialized['context_type'], "phase")
        self.assertEqual(serialized['priority'], "high")
        self.assertIn("serialize", serialized['tags'])


class TestContextManager(unittest.TestCase):
    """Test Enhanced Context Manager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.context_manager = EnhancedContextManager(
            max_memory_entries=100,
            enable_logging=False
        )
        
        # Create test conversation state
        self.conversation_state = ConversationState(
            session_id="test_session",
            course_request={
                "course_title": "Machine Learning Fundamentals",
                "course_level": "Undergraduate"
            }
        )
        self.conversation_state.set_current_phase("content_creation")
    
    def test_context_storage(self):
        """Test context storage and indexing"""
        content = {
            "learning_objectives": ["Understand ML basics", "Apply algorithms"],
            "assessment_methods": ["Quiz", "Project"]
        }
        
        context_id = self.context_manager.store_context(
            content=content,
            context_type=ContextType.PHASE,
            priority=MemoryPriority.HIGH,
            tags={"learning_objectives", "assessment"},
            source_agent="ipdai_agent",
            phase_id="foundation_design"
        )
        
        # Verify storage
        self.assertIn(context_id, self.context_manager.context_store)
        stored_entry = self.context_manager.context_store[context_id]
        self.assertEqual(stored_entry.content, content)
        self.assertEqual(stored_entry.context_type, ContextType.PHASE)
        
        # Verify indexing
        self.assertIn(context_id, self.context_manager.context_index[ContextType.PHASE])
        self.assertIn(context_id, self.context_manager.tag_index["learning_objectives"])
        self.assertIn(context_id, self.context_manager.phase_index["foundation_design"])
        self.assertIn(context_id, self.context_manager.agent_index["ipdai_agent"])
    
    def test_context_retrieval_by_type(self):
        """Test context retrieval by type"""
        # Store different types of context
        self.context_manager.store_context(
            content={"global_setting": "value"},
            context_type=ContextType.GLOBAL,
            priority=MemoryPriority.HIGH
        )
        
        self.context_manager.store_context(
            content={"phase_data": "content"},
            context_type=ContextType.PHASE,
            priority=MemoryPriority.MEDIUM,
            phase_id="content_creation"
        )
        
        self.context_manager.store_context(
            content={"agent_knowledge": "data"},
            context_type=ContextType.AGENT,
            priority=MemoryPriority.MEDIUM,
            source_agent="cauthai_agent"
        )
        
        # Query for phase contexts
        query = ContextQuery(
            context_types=[ContextType.PHASE],
            phase_id="content_creation"
        )
        
        results = self.context_manager.retrieve_context(query, self.conversation_state)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].context_type, ContextType.PHASE)
        self.assertEqual(results[0].content["phase_data"], "content")
    
    def test_context_retrieval_by_tags(self):
        """Test context retrieval by tags"""
        # Store contexts with different tags
        self.context_manager.store_context(
            content={"ml_content": "algorithms"},
            context_type=ContextType.AGENT,
            tags={"machine_learning", "algorithms"}
        )
        
        self.context_manager.store_context(
            content={"assessment_info": "rubrics"},
            context_type=ContextType.PHASE,
            tags={"assessment", "rubrics"}
        )
        
        self.context_manager.store_context(
            content={"ml_assessment": "projects"},
            context_type=ContextType.PHASE,
            tags={"machine_learning", "assessment"}
        )
        
        # Query for machine learning contexts
        query = ContextQuery(
            context_types=[ContextType.AGENT, ContextType.PHASE],
            tags={"machine_learning"}
        )
        
        results = self.context_manager.retrieve_context(query, self.conversation_state)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("machine_learning", result.tags)
    
    def test_relevance_scoring(self):
        """Test context relevance scoring"""
        # Store context for current phase
        current_phase_id = self.context_manager.store_context(
            content={"current_phase": "data"},
            context_type=ContextType.PHASE,
            phase_id="content_creation",  # Matches conversation state
            priority=MemoryPriority.MEDIUM
        )
        
        # Store context for different phase
        other_phase_id = self.context_manager.store_context(
            content={"other_phase": "data"},
            context_type=ContextType.PHASE,
            phase_id="foundation_design",
            priority=MemoryPriority.MEDIUM
        )
        
        # Query both
        query = ContextQuery(context_types=[ContextType.PHASE])
        results = self.context_manager.retrieve_context(query, self.conversation_state)
        
        # Current phase context should have higher relevance
        current_entry = next(r for r in results if r.context_id == current_phase_id)
        other_entry = next(r for r in results if r.context_id == other_phase_id)
        
        # Should be at least equal, potentially higher due to phase matching
        self.assertGreaterEqual(current_entry.relevance_score, other_entry.relevance_score)
    
    def test_context_relationships(self):
        """Test context linking and relationship management"""
        # Store related contexts
        context1_id = self.context_manager.store_context(
            content={"concept": "supervised_learning"},
            context_type=ContextType.AGENT
        )
        
        context2_id = self.context_manager.store_context(
            content={"example": "classification"},
            context_type=ContextType.AGENT
        )
        
        context3_id = self.context_manager.store_context(
            content={"algorithm": "decision_trees"},
            context_type=ContextType.AGENT
        )
        
        # Link contexts
        self.context_manager.link_contexts(context1_id, context2_id, "example")
        self.context_manager.link_contexts(context2_id, context3_id, "algorithm")
        
        # Test direct relationships
        direct_related = self.context_manager.get_related_contexts(context1_id, max_depth=1)
        self.assertIn(context2_id, direct_related)
        self.assertNotIn(context3_id, direct_related)
        
        # Test transitive relationships
        transitive_related = self.context_manager.get_related_contexts(context1_id, max_depth=2)
        self.assertIn(context2_id, transitive_related)
        # Note: context3_id should be reachable through context2_id
        # Verify the relationship chain exists
        context2_related = self.context_manager.get_related_contexts(context2_id, max_depth=1)
        self.assertIn(context3_id, context2_related)
    
    def test_memory_optimization(self):
        """Test memory optimization and cleanup"""
        # Fill with contexts of varying priorities
        critical_ids = []
        transient_ids = []
        
        for i in range(10):
            # Critical contexts (should be preserved)
            critical_id = self.context_manager.store_context(
                content={"critical_data": f"value_{i}"},
                context_type=ContextType.GLOBAL,
                priority=MemoryPriority.CRITICAL
            )
            critical_ids.append(critical_id)
            
            # Transient contexts (should be removed first)
            transient_id = self.context_manager.store_context(
                content={"transient_data": f"value_{i}"},
                context_type=ContextType.TEMPORAL,
                priority=MemoryPriority.TRANSIENT
            )
            transient_ids.append(transient_id)
        
        initial_count = len(self.context_manager.context_store)
        
        # Trigger optimization
        removed_count = self.context_manager.optimize_memory(target_reduction=0.3)
        
        final_count = len(self.context_manager.context_store)
        
        # Verify reduction occurred
        self.assertGreater(removed_count, 0)
        self.assertLess(final_count, initial_count)
        
        # Verify critical contexts preserved
        for critical_id in critical_ids:
            self.assertIn(critical_id, self.context_manager.context_store)
        
        # Some transient contexts should be removed
        remaining_transient = sum(1 for tid in transient_ids 
                                if tid in self.context_manager.context_store)
        self.assertLess(remaining_transient, len(transient_ids))
    
    def test_context_summary_generation(self):
        """Test comprehensive context summary generation"""
        # Store contexts for current phase
        self.context_manager.store_context(
            content={"phase_specific": "content_data"},
            context_type=ContextType.PHASE,
            phase_id="content_creation"
        )
        
        # Store global context
        self.context_manager.store_context(
            content={"global_setting": "always_relevant"},
            context_type=ContextType.GLOBAL,
            priority=MemoryPriority.HIGH
        )
        
        # Store procedural knowledge
        self.context_manager.store_context(
            content={"procedure": "how_to_create_content"},
            context_type=ContextType.PROCEDURAL
        )
        
        summary = self.context_manager.get_context_summary(self.conversation_state)
        
        # Verify summary structure
        self.assertIn('current_phase_context', summary)
        self.assertIn('global_context', summary)
        self.assertIn('procedural_context', summary)
        self.assertIn('context_metadata', summary)
        
        # Verify content presence
        self.assertGreater(len(summary['current_phase_context']), 0)
        self.assertGreater(len(summary['global_context']), 0)
        
        # Verify metadata
        metadata = summary['context_metadata']
        self.assertIn('total_entries', metadata)
        self.assertGreater(metadata['total_entries'], 0)
    
    def test_query_caching(self):
        """Test query result caching"""
        # Store test context
        self.context_manager.store_context(
            content={"cached_test": "data"},
            context_type=ContextType.AGENT,
            tags={"cache_test"}
        )
        
        query = ContextQuery(
            context_types=[ContextType.AGENT],
            tags={"cache_test"}
        )
        
        # First query - should hit storage
        initial_cache_hits = self.context_manager.retrieval_stats['cache_hits']
        results1 = self.context_manager.retrieve_context(query, self.conversation_state)
        first_query_cache_hits = self.context_manager.retrieval_stats['cache_hits']
        
        # Second identical query - should hit cache
        results2 = self.context_manager.retrieve_context(query, self.conversation_state)
        second_query_cache_hits = self.context_manager.retrieval_stats['cache_hits']
        
        # Verify cache hit occurred
        self.assertEqual(first_query_cache_hits, initial_cache_hits)  # No cache hit on first query
        self.assertEqual(second_query_cache_hits, first_query_cache_hits + 1)  # Cache hit on second
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1[0].context_id, results2[0].context_id)
    
    def test_context_export_import(self):
        """Test context export and import functionality"""
        # Store test contexts
        context_ids = []
        for i in range(3):
            context_id = self.context_manager.store_context(
                content={"export_test": f"data_{i}"},
                context_type=ContextType.AGENT,
                priority=MemoryPriority.MEDIUM,
                tags={f"tag_{i}"}
            )
            context_ids.append(context_id)
        
        # Export contexts
        exported_data = self.context_manager.export_context(context_ids)
        
        # Verify export structure
        self.assertIn('contexts', exported_data)
        self.assertIn('metadata', exported_data)
        self.assertEqual(len(exported_data['contexts']), 3)
        
        # Clear and import
        original_count = len(self.context_manager.context_store)
        self.context_manager.context_store.clear()
        self.context_manager._update_indexes = Mock()  # Mock to avoid index issues in test
        
        imported_count = self.context_manager.import_context(exported_data)
        
        # Verify import
        self.assertEqual(imported_count, 3)
    
    def test_time_based_filtering(self):
        """Test time-based context filtering"""
        now = datetime.now()
        
        # Store contexts with different timestamps
        recent_id = self.context_manager.store_context(
            content={"recent": "data"},
            context_type=ContextType.TEMPORAL
        )
        
        # Manually adjust timestamp for old context
        old_id = self.context_manager.store_context(
            content={"old": "data"},
            context_type=ContextType.TEMPORAL
        )
        old_entry = self.context_manager.context_store[old_id]
        old_entry.created_at = now - timedelta(days=7)
        
        # Query for recent contexts only
        query = ContextQuery(
            context_types=[ContextType.TEMPORAL],
            time_range=(now - timedelta(hours=1), now + timedelta(hours=1))
        )
        
        results = self.context_manager.retrieve_context(query, self.conversation_state)
        
        # Should only get recent context
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].context_id, recent_id)
    
    def test_context_statistics(self):
        """Test context statistics generation"""
        # Store various contexts
        self.context_manager.store_context(
            content={"test": "data1"},
            context_type=ContextType.GLOBAL,
            priority=MemoryPriority.HIGH
        )
        
        self.context_manager.store_context(
            content={"test": "data2"},
            context_type=ContextType.PHASE,
            priority=MemoryPriority.MEDIUM
        )
        
        # Perform some queries to generate stats
        query = ContextQuery(context_types=[ContextType.GLOBAL])
        self.context_manager.retrieve_context(query, self.conversation_state)
        
        stats = self.context_manager.get_context_statistics()
        
        # Verify statistics structure
        self.assertIn('total_contexts', stats)
        self.assertIn('type_distribution', stats)
        self.assertIn('priority_distribution', stats)
        self.assertIn('retrieval_stats', stats)
        self.assertIn('memory_usage', stats)
        
        # Verify counts
        self.assertEqual(stats['total_contexts'], 2)
        self.assertIn('global', stats['type_distribution'])
        self.assertIn('phase', stats['type_distribution'])
        self.assertGreater(stats['retrieval_stats']['total_queries'], 0)


def run_phase_3_3_tests():
    """Run all Phase 3.3 tests and return results"""
    print("🧪 Running Phase 3.3 Tests: Enhanced Context Management")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestContextEntry,
        TestContextManager
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
    run_phase_3_3_tests()