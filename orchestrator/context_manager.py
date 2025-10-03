"""
HAILEI Enhanced Context Management System

Advanced context management with memory optimization, intelligent caching,
and context-aware information retrieval for conversational workflows.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque
import threading

from .conversation_state import ConversationState, AgentOutput, PhaseStatus


class ContextType(Enum):
    """Types of context information"""
    GLOBAL = "global"
    PHASE = "phase"
    AGENT = "agent"
    USER = "user"
    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"


class MemoryPriority(Enum):
    """Memory priority levels for retention"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRANSIENT = "transient"


@dataclass
class ContextEntry:
    """Individual context entry with metadata"""
    context_id: str
    context_type: ContextType
    content: Dict[str, Any]
    priority: MemoryPriority
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    relevance_score: float = 1.0
    tags: Set[str] = field(default_factory=set)
    source_agent: Optional[str] = None
    phase_id: Optional[str] = None
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_retention_score(self) -> float:
        """Calculate score for memory retention decisions"""
        # Base score from priority
        priority_weights = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.6,
            MemoryPriority.LOW: 0.4,
            MemoryPriority.TRANSIENT: 0.2
        }
        
        base_score = priority_weights[self.priority]
        
        # Recent access bonus
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        recency_factor = max(0, 1.0 - (hours_since_access / 168))  # Decay over a week
        
        # Access frequency bonus
        frequency_factor = min(1.0, self.access_count / 10.0)
        
        # Relevance factor
        relevance_factor = self.relevance_score
        
        return base_score * 0.4 + recency_factor * 0.3 + frequency_factor * 0.2 + relevance_factor * 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'context_id': self.context_id,
            'context_type': self.context_type.value,
            'content': self.content,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'relevance_score': self.relevance_score,
            'tags': list(self.tags),
            'source_agent': self.source_agent,
            'phase_id': self.phase_id
        }


@dataclass
class ContextQuery:
    """Query for context retrieval"""
    context_types: List[ContextType]
    tags: Optional[Set[str]] = None
    phase_id: Optional[str] = None
    agent_id: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    min_relevance: float = 0.0
    max_results: int = 50
    include_transient: bool = False


class EnhancedContextManager:
    """
    Advanced context management system for HAILEI conversational workflows.
    
    Features:
    - Intelligent memory optimization
    - Context-aware retrieval
    - Automatic relevance scoring
    - Memory pressure management
    - Cross-phase context linking
    - Semantic context clustering
    """
    
    def __init__(
        self,
        max_memory_entries: int = 10000,
        memory_cleanup_threshold: float = 0.8,
        enable_logging: bool = True
    ):
        """Initialize enhanced context manager"""
        self.max_memory_entries = max_memory_entries
        self.memory_cleanup_threshold = memory_cleanup_threshold
        
        # Context storage
        self.context_store: Dict[str, ContextEntry] = {}
        self.context_index: Dict[ContextType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.phase_index: Dict[str, Set[str]] = defaultdict(set)
        self.agent_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Memory management
        self.access_history: deque = deque(maxlen=1000)
        self.cleanup_lock = threading.Lock()
        self.last_cleanup = datetime.now()
        
        # Context relationships
        self.context_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.semantic_clusters: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.retrieval_stats: Dict[str, Any] = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0,
            'memory_usage': 0
        }
        
        # Caching for frequent queries
        self.query_cache: Dict[str, Tuple[List[ContextEntry], datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger('hailei_context_manager')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logging.getLogger('null')
            self.logger.addHandler(logging.NullHandler())
    
    def store_context(
        self,
        content: Dict[str, Any],
        context_type: ContextType,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: Optional[Set[str]] = None,
        source_agent: Optional[str] = None,
        phase_id: Optional[str] = None,
        context_id: Optional[str] = None
    ) -> str:
        """
        Store context information with intelligent indexing.
        
        Args:
            content: Context content to store
            context_type: Type of context
            priority: Memory priority level
            tags: Tags for categorization
            source_agent: Agent that created this context
            phase_id: Phase this context belongs to
            context_id: Optional custom context ID
            
        Returns:
            str: Generated or provided context ID
        """
        # Generate context ID if not provided
        if not context_id:
            content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:8]
            context_id = f"{context_type.value}_{content_hash}_{datetime.now().timestamp()}"
        
        # Create context entry
        entry = ContextEntry(
            context_id=context_id,
            context_type=context_type,
            content=content,
            priority=priority,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags=tags or set(),
            source_agent=source_agent,
            phase_id=phase_id
        )
        
        # Store and index
        self.context_store[context_id] = entry
        self._update_indexes(entry)
        
        # Check memory pressure
        if len(self.context_store) > self.max_memory_entries * self.memory_cleanup_threshold:
            self._trigger_memory_cleanup()
        
        # Invalidate relevant caches
        self._invalidate_cache_for_entry(entry)
        
        self.logger.info(f"Stored context: {context_id} ({context_type.value}, priority: {priority.value})")
        return context_id
    
    def retrieve_context(
        self,
        query: ContextQuery,
        conversation_state: Optional[ConversationState] = None
    ) -> List[ContextEntry]:
        """
        Retrieve context based on intelligent query matching.
        
        Args:
            query: Context query specification
            conversation_state: Current conversation state for relevance scoring
            
        Returns:
            List of matching context entries, sorted by relevance
        """
        start_time = datetime.now()
        self.retrieval_stats['total_queries'] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        if cache_key in self.query_cache:
            cached_results, cache_time = self.query_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                self.retrieval_stats['cache_hits'] += 1
                return cached_results
        
        # Find matching entries
        candidate_ids = self._find_candidate_entries(query)
        matching_entries = []
        
        for context_id in candidate_ids:
            entry = self.context_store[context_id]
            
            # Apply filters
            if not self._entry_matches_query(entry, query):
                continue
            
            # Update access statistics
            entry.update_access()
            
            # Calculate relevance if conversation state provided
            if conversation_state:
                entry.relevance_score = self._calculate_relevance(entry, conversation_state)
            
            matching_entries.append(entry)
        
        # Sort by relevance and priority
        matching_entries.sort(key=lambda e: (e.relevance_score, e.priority.value), reverse=True)
        
        # Limit results
        result_entries = matching_entries[:query.max_results]
        
        # Cache results
        self.query_cache[cache_key] = (result_entries, datetime.now())
        
        # Update stats
        retrieval_time = (datetime.now() - start_time).total_seconds()
        self.retrieval_stats['avg_retrieval_time'] = (
            self.retrieval_stats['avg_retrieval_time'] * 0.9 + retrieval_time * 0.1
        )
        
        self.logger.info(f"Retrieved {len(result_entries)} context entries in {retrieval_time:.3f}s")
        return result_entries
    
    def update_context_relevance(
        self,
        context_id: str,
        new_relevance: float,
        reason: Optional[str] = None
    ):
        """Update context relevance based on usage patterns"""
        if context_id in self.context_store:
            entry = self.context_store[context_id]
            old_relevance = entry.relevance_score
            entry.relevance_score = new_relevance
            
            self.logger.info(f"Updated relevance for {context_id}: {old_relevance:.3f} -> {new_relevance:.3f}")
            if reason:
                self.logger.info(f"Reason: {reason}")
    
    def link_contexts(self, context_id1: str, context_id2: str, relationship_type: str = "related"):
        """Create relationships between contexts"""
        self.context_relationships[context_id1].add(context_id2)
        self.context_relationships[context_id2].add(context_id1)
        
        self.logger.info(f"Linked contexts: {context_id1} <-> {context_id2} ({relationship_type})")
    
    def get_related_contexts(self, context_id: str, max_depth: int = 2) -> Set[str]:
        """Get contexts related to a given context"""
        related = set()
        to_explore = {context_id}
        
        for depth in range(max_depth):
            if not to_explore:
                break
            
            current_level = set()
            for ctx_id in to_explore:
                if ctx_id in self.context_relationships:
                    current_level.update(self.context_relationships[ctx_id])
            
            related.update(current_level)
            to_explore = current_level - related
        
        return related - {context_id}
    
    def optimize_memory(self, target_reduction: float = 0.2) -> int:
        """
        Optimize memory usage by removing low-value contexts.
        
        Args:
            target_reduction: Target reduction as a fraction of current size
            
        Returns:
            int: Number of entries removed
        """
        with self.cleanup_lock:
            if not self.context_store:
                return 0
            
            current_count = len(self.context_store)
            target_count = int(current_count * (1 - target_reduction))
            
            # Calculate retention scores for all entries
            retention_scores = []
            for context_id, entry in self.context_store.items():
                score = entry.calculate_retention_score()
                retention_scores.append((score, context_id))
            
            # Sort by retention score (lowest first for removal)
            retention_scores.sort()
            
            # Remove lowest scoring entries
            entries_to_remove = retention_scores[:current_count - target_count]
            removed_count = 0
            
            for score, context_id in entries_to_remove:
                if self._can_remove_entry(context_id):
                    self._remove_context_entry(context_id)
                    removed_count += 1
            
            self.last_cleanup = datetime.now()
            self.logger.info(f"Memory optimization complete: removed {removed_count} entries")
            
            return removed_count
    
    def get_context_summary(self, conversation_state: ConversationState) -> Dict[str, Any]:
        """Get comprehensive context summary for agents"""
        
        # Recent context from current phase
        current_phase_query = ContextQuery(
            context_types=[ContextType.PHASE, ContextType.AGENT],
            phase_id=conversation_state.current_phase,
            max_results=10
        )
        recent_phase_context = self.retrieve_context(current_phase_query, conversation_state)
        
        # Global context that's always relevant
        global_query = ContextQuery(
            context_types=[ContextType.GLOBAL, ContextType.USER],
            min_relevance=0.7,
            max_results=5
        )
        global_context = self.retrieve_context(global_query, conversation_state)
        
        # Cross-phase procedural knowledge
        procedural_query = ContextQuery(
            context_types=[ContextType.PROCEDURAL],
            max_results=5
        )
        procedural_context = self.retrieve_context(procedural_query, conversation_state)
        
        return {
            'current_phase_context': [entry.content for entry in recent_phase_context],
            'global_context': [entry.content for entry in global_context],
            'procedural_context': [entry.content for entry in procedural_context],
            'context_metadata': {
                'total_entries': len(self.context_store),
                'phase_entries': len(self.phase_index.get(conversation_state.current_phase or '', set())),
                'last_cleanup': self.last_cleanup.isoformat(),
                'memory_usage': self._calculate_memory_usage()
            }
        }
    
    def export_context(self, context_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export context for persistence or transfer"""
        if context_ids is None:
            context_ids = list(self.context_store.keys())
        
        exported_data = {
            'contexts': {},
            'relationships': {},
            'clusters': {},
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_entries': len(context_ids)
            }
        }
        
        for context_id in context_ids:
            if context_id in self.context_store:
                exported_data['contexts'][context_id] = self.context_store[context_id].to_dict()
                
                if context_id in self.context_relationships:
                    exported_data['relationships'][context_id] = list(self.context_relationships[context_id])
        
        return exported_data
    
    def import_context(self, exported_data: Dict[str, Any]) -> int:
        """Import previously exported context data"""
        imported_count = 0
        
        # Import contexts
        for context_id, context_data in exported_data.get('contexts', {}).items():
            try:
                entry = self._context_entry_from_dict(context_data)
                self.context_store[context_id] = entry
                self._update_indexes(entry)
                imported_count += 1
            except Exception as e:
                self.logger.error(f"Failed to import context {context_id}: {e}")
        
        # Import relationships
        for context_id, related_ids in exported_data.get('relationships', {}).items():
            if context_id in self.context_store:
                self.context_relationships[context_id] = set(related_ids)
        
        self.logger.info(f"Imported {imported_count} context entries")
        return imported_count
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get comprehensive context management statistics"""
        type_distribution = defaultdict(int)
        priority_distribution = defaultdict(int)
        
        for entry in self.context_store.values():
            type_distribution[entry.context_type.value] += 1
            priority_distribution[entry.priority.value] += 1
        
        return {
            'total_contexts': len(self.context_store),
            'type_distribution': dict(type_distribution),
            'priority_distribution': dict(priority_distribution),
            'memory_usage': self._calculate_memory_usage(),
            'retrieval_stats': self.retrieval_stats.copy(),
            'cache_size': len(self.query_cache),
            'relationships_count': sum(len(rels) for rels in self.context_relationships.values()),
            'last_cleanup': self.last_cleanup.isoformat(),
            'indexes': {
                'tags': len(self.tag_index),
                'phases': len(self.phase_index),
                'agents': len(self.agent_index)
            }
        }
    
    def _update_indexes(self, entry: ContextEntry):
        """Update all indexes for a context entry"""
        self.context_index[entry.context_type].add(entry.context_id)
        
        for tag in entry.tags:
            self.tag_index[tag].add(entry.context_id)
        
        if entry.phase_id:
            self.phase_index[entry.phase_id].add(entry.context_id)
        
        if entry.source_agent:
            self.agent_index[entry.source_agent].add(entry.context_id)
    
    def _find_candidate_entries(self, query: ContextQuery) -> Set[str]:
        """Find candidate entries based on indexes"""
        candidates = set()
        
        # Start with context type filtering
        for context_type in query.context_types:
            candidates.update(self.context_index[context_type])
        
        # Narrow down by tags
        if query.tags:
            tag_matches = set()
            for tag in query.tags:
                tag_matches.update(self.tag_index[tag])
            candidates &= tag_matches
        
        # Filter by phase
        if query.phase_id:
            candidates &= self.phase_index[query.phase_id]
        
        # Filter by agent
        if query.agent_id:
            candidates &= self.agent_index[query.agent_id]
        
        return candidates
    
    def _entry_matches_query(self, entry: ContextEntry, query: ContextQuery) -> bool:
        """Check if entry matches query filters"""
        # Time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not (start_time <= entry.created_at <= end_time):
                return False
        
        # Relevance filter
        if entry.relevance_score < query.min_relevance:
            return False
        
        # Transient filter
        if not query.include_transient and entry.priority == MemoryPriority.TRANSIENT:
            return False
        
        return True
    
    def _calculate_relevance(self, entry: ContextEntry, conversation_state: ConversationState) -> float:
        """Calculate context relevance to current conversation state"""
        relevance = entry.relevance_score
        
        # Phase relevance
        if entry.phase_id == conversation_state.current_phase:
            relevance += 0.3
        
        # Recent creation bonus
        hours_old = (datetime.now() - entry.created_at).total_seconds() / 3600
        if hours_old < 1:
            relevance += 0.2
        elif hours_old < 24:
            relevance += 0.1
        
        # Tag matching with conversation context
        conversation_tags = self._extract_conversation_tags(conversation_state)
        tag_overlap = len(entry.tags & conversation_tags)
        if tag_overlap > 0:
            relevance += tag_overlap * 0.1
        
        return min(1.0, relevance)
    
    def _extract_conversation_tags(self, conversation_state: ConversationState) -> Set[str]:
        """Extract relevant tags from conversation state"""
        tags = set()
        
        # Add phase as tag
        if conversation_state.current_phase:
            tags.add(conversation_state.current_phase)
        
        # Add course subject tags
        course_request = conversation_state.course_request
        if 'course_title' in course_request:
            # Simple keyword extraction
            title_words = course_request['course_title'].lower().split()
            tags.update(word for word in title_words if len(word) > 3)
        
        return tags
    
    def _trigger_memory_cleanup(self):
        """Trigger automatic memory cleanup in background"""
        if datetime.now() - self.last_cleanup < timedelta(minutes=5):
            return  # Don't cleanup too frequently
        
        # Run cleanup in background thread to avoid blocking
        def cleanup_thread():
            try:
                self.optimize_memory(target_reduction=0.1)
            except Exception as e:
                self.logger.error(f"Memory cleanup failed: {e}")
        
        threading.Thread(target=cleanup_thread, daemon=True).start()
    
    def _can_remove_entry(self, context_id: str) -> bool:
        """Check if context entry can be safely removed"""
        entry = self.context_store.get(context_id)
        if not entry:
            return True
        
        # Never remove critical priority
        if entry.priority == MemoryPriority.CRITICAL:
            return False
        
        # Don't remove recently accessed high priority
        if entry.priority == MemoryPriority.HIGH:
            hours_since_access = (datetime.now() - entry.last_accessed).total_seconds() / 3600
            if hours_since_access < 24:
                return False
        
        return True
    
    def _remove_context_entry(self, context_id: str):
        """Remove context entry and update indexes"""
        if context_id not in self.context_store:
            return
        
        entry = self.context_store[context_id]
        
        # Remove from indexes
        self.context_index[entry.context_type].discard(context_id)
        
        for tag in entry.tags:
            self.tag_index[tag].discard(context_id)
            if not self.tag_index[tag]:
                del self.tag_index[tag]
        
        if entry.phase_id:
            self.phase_index[entry.phase_id].discard(context_id)
            if not self.phase_index[entry.phase_id]:
                del self.phase_index[entry.phase_id]
        
        if entry.source_agent:
            self.agent_index[entry.source_agent].discard(context_id)
            if not self.agent_index[entry.source_agent]:
                del self.agent_index[entry.source_agent]
        
        # Remove relationships
        if context_id in self.context_relationships:
            for related_id in self.context_relationships[context_id]:
                self.context_relationships[related_id].discard(context_id)
            del self.context_relationships[context_id]
        
        # Remove from store
        del self.context_store[context_id]
    
    def _invalidate_cache_for_entry(self, entry: ContextEntry):
        """Invalidate relevant cache entries when new context is added"""
        # Simple cache invalidation - remove entries that might be affected
        keys_to_remove = []
        for cache_key in self.query_cache:
            # Heuristic: if cache key contains entry's type or tags, invalidate
            if entry.context_type.value in cache_key:
                keys_to_remove.append(cache_key)
            elif any(tag in cache_key for tag in entry.tags):
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.query_cache[key]
    
    def _generate_cache_key(self, query: ContextQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            "_".join(ct.value for ct in query.context_types),
            "_".join(sorted(query.tags)) if query.tags else "no_tags",
            query.phase_id or "no_phase",
            query.agent_id or "no_agent",
            str(query.min_relevance),
            str(query.max_results),
            str(query.include_transient)
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _calculate_memory_usage(self) -> int:
        """Calculate approximate memory usage in bytes"""
        # Rough estimation
        total_size = 0
        for entry in self.context_store.values():
            # Estimate entry size
            content_size = len(json.dumps(entry.content).encode())
            metadata_size = 200  # Approximate metadata size
            total_size += content_size + metadata_size
        
        return total_size
    
    def _context_entry_from_dict(self, data: Dict[str, Any]) -> ContextEntry:
        """Create ContextEntry from dictionary data"""
        return ContextEntry(
            context_id=data['context_id'],
            context_type=ContextType(data['context_type']),
            content=data['content'],
            priority=MemoryPriority(data['priority']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data['access_count'],
            relevance_score=data['relevance_score'],
            tags=set(data['tags']),
            source_agent=data.get('source_agent'),
            phase_id=data.get('phase_id')
        )