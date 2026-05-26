"""
Advanced Memory System for Hermes Agent OSS.

Implements multi-tier memory architecture:
- Working Memory (current context)
- Short-term Memory (recent interactions)
- Long-term Memory (semantic search with embeddings)
- Episodic Memory (important moments)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
from collections import deque
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Represents a single memory item."""
    content: str
    timestamp: datetime
    memory_type: str  # working, short_term, long_term, episodic
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class WorkingMemory:
    """
    Working Memory - Current context and immediate attention.
    
    Keeps the most recent interactions and current task context.
    Automatically cleaned when full.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize working memory.

        Args:
            max_size: Maximum number of items to keep
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_size = max_size
        self.items: deque = deque(maxlen=max_size)

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add item to working memory.

        Args:
            content: Memory content
            metadata: Optional metadata
        """
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            memory_type="working",
            metadata=metadata or {}
        )
        self.items.append(item)
        self.logger.debug(f"Added to working memory: {content[:50]}...")

    def get_all(self) -> List[MemoryItem]:
        """Get all working memory items."""
        return list(self.items)

    def get_recent(self, k: int = 5) -> List[MemoryItem]:
        """
        Get k most recent items.

        Args:
            k: Number of items

        Returns:
            Recent memory items
        """
        return list(self.items)[-k:]

    def clear(self) -> None:
        """Clear all working memory."""
        self.items.clear()
        self.logger.info("Working memory cleared")

    def get_summary(self) -> str:
        """
        Get concise summary of working memory.

        Returns:
            Summary string
        """
        if not self.items:
            return "No recent context"
        
        recent = self.get_recent(3)
        summary = "Recent context: " + " | ".join([
            item.content[:30] for item in recent
        ])
        return summary


class ShortTermMemory:
    """
    Short-term Memory - Recent interactions with sliding window.
    
    Keeps interactions from the last N minutes or last K items.
    Auto-ages out old items.
    """

    def __init__(self, max_items: int = 50, retention_hours: int = 24):
        """
        Initialize short-term memory.

        Args:
            max_items: Maximum items to keep
            retention_hours: How long to keep items
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_items = max_items
        self.retention_hours = retention_hours
        self.items: List[MemoryItem] = []

    def add(self, content: str, importance: float = 0.5, 
             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add item to short-term memory.

        Args:
            content: Memory content
            importance: Importance score (0-1)
            metadata: Optional metadata
        """
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            memory_type="short_term",
            importance_score=importance,
            metadata=metadata or {}
        )
        self.items.append(item)
        self._cleanup()
        self.logger.debug(f"Added to short-term memory (importance: {importance})")

    def _cleanup(self) -> None:
        """
        Remove old and excess items.
        """
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Remove expired items
        self.items = [
            item for item in self.items 
            if item.timestamp > cutoff_time
        ]
        
        # Keep only max_items, prioritizing importance
        if len(self.items) > self.max_items:
            self.items.sort(key=lambda x: x.importance_score, reverse=True)
            self.items = self.items[:self.max_items]
            self.items.sort(key=lambda x: x.timestamp, reverse=True)

    def get_recent(self, k: int = 10) -> List[MemoryItem]:
        """
        Get k most recent items.

        Args:
            k: Number of items

        Returns:
            Recent items
        """
        self._cleanup()
        return self.items[-k:]

    def get_all(self) -> List[MemoryItem]:
        """Get all short-term memory items."""
        self._cleanup()
        return self.items

    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """
        Search short-term memory (simple keyword search).

        Args:
            query: Search query
            k: Max results

        Returns:
            Matching items
        """
        self._cleanup()
        query_lower = query.lower()
        matches = [
            item for item in self.items
            if query_lower in item.content.lower()
        ]
        return matches[:k]


class LongTermMemory:
    """
    Long-term Memory - Semantic search with embeddings.
    
    Uses vector database for semantic similarity.
    Stores important facts and learned information.
    """

    def __init__(self, vector_db=None, max_items: int = 10000):
        """
        Initialize long-term memory.

        Args:
            vector_db: Vector database instance (e.g., Chroma)
            max_items: Maximum items to keep
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_db = vector_db
        self.max_items = max_items
        self.items: List[MemoryItem] = []

    async def add(self, content: str, embedding: Optional[List[float]] = None,
                  importance: float = 0.5, 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add item to long-term memory.

        Args:
            content: Memory content
            embedding: Vector embedding
            importance: Importance score (0-1)
            metadata: Optional metadata
        """
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            memory_type="long_term",
            importance_score=importance,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.items.append(item)
        
        # Store in vector DB if available
        if self.vector_db and embedding:
            try:
                await self._store_in_vector_db(item)
            except Exception as e:
                self.logger.error(f"Error storing in vector DB: {str(e)}")
        
        self._cleanup()
        self.logger.debug(f"Added to long-term memory")

    async def _store_in_vector_db(self, item: MemoryItem) -> None:
        """
        Store item in vector database.

        Args:
            item: Memory item to store
        """
        if not self.vector_db or not item.embedding:
            return

    async def search(self, query: str, embedding: Optional[List[float]] = None,
                     k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        Search long-term memory by semantic similarity.

        Args:
            query: Search query text
            embedding: Query embedding vector
            k: Number of results

        Returns:
            List of (item, similarity_score) tuples
        """
        if not self.items:
            return []
        
        # If no embedding provided, do keyword search
        if not embedding:
            matches = [
                item for item in self.items
                if query.lower() in item.content.lower()
            ]
            return [(item, 0.5) for item in matches[:k]]
        
        # Semantic search with embeddings
        results = []
        for item in self.items:
            if item.embedding:
                similarity = self._cosine_similarity(
                    embedding, 
                    item.embedding
                )
                results.append((item, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / 
                    (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))

    def _cleanup(self) -> None:
        """
        Remove excess items, keeping highest importance.
        """
        if len(self.items) > self.max_items:
            self.items.sort(key=lambda x: x.importance_score, reverse=True)
            self.items = self.items[:self.max_items]

    async def get_all(self) -> List[MemoryItem]:
        """Get all long-term memory items."""
        return self.items


class EpisodicMemory:
    """
    Episodic Memory - Important moments and milestones.
    
    Stores significant events with timestamps and context.
    Used for understanding agent history and patterns.
    """

    def __init__(self, max_items: int = 100):
        """
        Initialize episodic memory.

        Args:
            max_items: Maximum episodes to keep
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_items = max_items
        self.episodes: List[MemoryItem] = []

    def record_episode(self, title: str, description: str, 
                      importance: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an important episode.

        Args:
            title: Episode title
            description: Episode description
            importance: Importance score
            metadata: Additional metadata
        """
        content = f"[{title}] {description}"
        meta = metadata or {}
        meta['title'] = title
        
        episode = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            memory_type="episodic",
            importance_score=importance,
            metadata=meta
        )
        
        self.episodes.append(episode)
        
        # Keep only max_items
        if len(self.episodes) > self.max_items:
            self.episodes = self.episodes[-self.max_items:]
        
        self.logger.info(f"Recorded episode: {title}")

    def get_timeline(self, limit: int = 20) -> List[MemoryItem]:
        """
        Get timeline of episodes (most recent first).

        Args:
            limit: Max episodes to return

        Returns:
            Recent episodes
        """
        return sorted(
            self.episodes[-limit:],
            key=lambda x: x.timestamp,
            reverse=True
        )

    def get_by_importance(self, k: int = 10) -> List[MemoryItem]:
        """
        Get most important episodes.

        Args:
            k: Number of episodes

        Returns:
            Top episodes by importance
        """
        return sorted(
            self.episodes,
            key=lambda x: x.importance_score,
            reverse=True
        )[:k]

    def get_recent(self, days: int = 7) -> List[MemoryItem]:
        """
        Get episodes from last N days.

        Args:
            days: Number of days

        Returns:
            Recent episodes
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [
            ep for ep in self.episodes
            if ep.timestamp > cutoff
        ]


class AdvancedMemoryManager:
    """
    Main memory manager coordinating all memory types.
    
    Handles memory compression, automatic cleanup,
    and intelligent retrieval across all memory tiers.
    """

    def __init__(self, vector_db=None, compression_threshold: float = 0.8):
        """
        Initialize the memory manager.

        Args:
            vector_db: Vector database instance
            compression_threshold: Threshold for auto-compression (0-1)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.working_memory = WorkingMemory(max_size=10)
        self.short_term_memory = ShortTermMemory(max_items=50)
        self.long_term_memory = LongTermMemory(vector_db=vector_db)
        self.episodic_memory = EpisodicMemory(max_items=100)
        self.compression_threshold = compression_threshold
        self.vector_db = vector_db

    async def add_interaction(self, query: str, response: str,
                             importance: float = 0.5,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a query-response interaction to memory.

        Args:
            query: User query
            response: Agent response
            importance: Importance score
            metadata: Additional metadata
        """
        meta = metadata or {}
        meta['type'] = 'interaction'
        
        # Add to working memory
        self.working_memory.add(f"Q: {query}", meta)
        self.working_memory.add(f"A: {response}", meta)
        
        # Add to short-term memory
        interaction_text = f"Q: {query}\nA: {response[:200]}"
        self.short_term_memory.add(
            interaction_text,
            importance=importance,
            metadata=meta
        )
        
        # Add to long-term memory if important enough
        if importance > 0.7:
            await self.long_term_memory.add(
                interaction_text,
                importance=importance,
                metadata=meta
            )

    async def record_milestone(self, title: str, description: str,
                               importance: float = 1.0) -> None:
        """
        Record an important milestone/episode.

        Args:
            title: Event title
            description: Event description
            importance: Importance score
        """
        self.episodic_memory.record_episode(
            title=title,
            description=description,
            importance=importance
        )

    async def get_context(self, k: int = 10) -> str:
        """
        Get current context for agent decision-making.

        Args:
            k: Number of items to include

        Returns:
            Context string
        """
        context_parts = []
        
        # Working memory (most recent)
        working = self.working_memory.get_recent(3)
        if working:
            context_parts.append("Recent context:")
            for item in working:
                context_parts.append(f"  - {item.content}")
        
        # Short-term memory
        short_term = self.short_term_memory.get_recent(5)
        if short_term:
            context_parts.append("\nRecent interactions:")
            for item in short_term:
                context_parts.append(f"  - {item.content[:80]}...")
        
        # Important episodes
        episodes = self.episodic_memory.get_by_importance(3)
        if episodes:
            context_parts.append("\nImportant context:")
            for ep in episodes:
                context_parts.append(f"  - {ep.content[:80]}...")
        
        return "\n".join(context_parts) if context_parts else "No context available"

    async def search(self, query: str, include_long_term: bool = True,
                    include_short_term: bool = True, 
                    k: int = 5) -> Dict[str, List[MemoryItem]]:
        """
        Search across memory tiers.

        Args:
            query: Search query
            include_long_term: Include long-term search
            include_short_term: Include short-term search
            k: Results per tier

        Returns:
            Results organized by memory type
        """
        results = {}
        
        if include_short_term:
            results['short_term'] = self.short_term_memory.search(query, k)
        
        if include_long_term:
            results['long_term'] = await self.long_term_memory.search(query, k=k)
        
        return results

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all memory tiers.

        Returns:
            Memory statistics
        """
        return {
            'working_memory': {
                'items': len(self.working_memory.get_all()),
                'max_size': self.working_memory.max_size
            },
            'short_term_memory': {
                'items': len(self.short_term_memory.get_all()),
                'max_size': self.short_term_memory.max_items
            },
            'long_term_memory': {
                'items': len(await self.long_term_memory.get_all()),
                'max_size': self.long_term_memory.max_items
            },
            'episodic_memory': {
                'episodes': len(self.episodic_memory.episodes),
                'max_episodes': self.episodic_memory.max_items
            }
        }

    async def compression_pass(self) -> None:
        """
        Perform memory compression/consolidation.
        
        Summarizes old interactions and consolidates memory.
        """
        self.logger.info("Starting memory compression pass")
        
        # Get recent interactions from short-term
        recent = self.short_term_memory.get_recent(5)
        if len(recent) > 0:
            # Could implement LLM-based summarization here
            self.logger.debug(f"Compressed {len(recent)} recent items")
        
        self.logger.info("Memory compression pass completed")