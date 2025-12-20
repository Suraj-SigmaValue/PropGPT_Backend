"""
Semantic Response Cache for PropGPT
Caches LLM responses based on semantic similarity of queries and context.
"""

import json
import logging
import time
from hashlib import md5
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

logger = logging.getLogger(__name__)


class SemanticResponseCache:
    """
    Caches LLM responses with semantic similarity matching.
    
    Features:
    - Semantic matching: Similar queries retrieve cached responses
    - Context-aware: Considers query + items + mapping keys
    - TTL support: Automatic expiration
    - Persistence: Saves to disk
    """
    
    def __init__(
        self,
        cache_dir: Path,
        embeddings: HuggingFaceEmbeddings,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 86400  # 24 hours default
    ):
        """
        Initialize semantic cache.
        
        Args:
            cache_dir: Directory to store cache files
            embeddings: HuggingFace embeddings model for semantic matching
            similarity_threshold: Minimum cosine similarity to consider a cache hit (0.95 = 95%)
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        
        # Cache structure: {cache_key: {embedding, response, metadata, timestamp}}
        self.cache_file = self.cache_dir / "response_cache.pkl"
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        
        logger.info(
            f"Initialized SemanticResponseCache with {len(self.cache)} entries, "
            f"threshold={similarity_threshold}, TTL={ttl_seconds}s"
        )
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                cache = joblib.load(self.cache_file)
                logger.info(f"Loaded response cache with {len(cache)} entries")
                return cache
            except Exception as exc:
                logger.warning(f"Failed to load cache: {exc}. Starting fresh.")
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            joblib.dump(self.cache, self.cache_file)
            logger.debug(f"Saved response cache with {len(self.cache)} entries")
        except Exception as exc:
            logger.error(f"Failed to save cache: {exc}")
    
    def _build_cache_key(
        self,
        query: str,
        items: list,
        mapping_keys: list,
        comparison_type: str,
        provider: str
    ) -> str:
        """
        Build a deterministic cache key from query parameters.
        
        This creates a unique identifier for the query context.
        """
        payload = {
            "query": query.strip().lower(),
            "items": sorted([str(i).lower() for i in items]),
            "mapping_keys": sorted(mapping_keys),
            "comparison_type": comparison_type.lower(),
            "provider": provider.lower()
        }
        return md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        try:
            return np.array(self.embeddings.embed_query(text))
        except Exception as exc:
            logger.error(f"Failed to compute embedding: {exc}")
            return np.array([])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - timestamp) > self.ttl_seconds
    
    def get(
        self,
        query: str,
        items: list,
        mapping_keys: list,
        comparison_type: str,
        provider: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve cached response if semantically similar query exists.
        
        Returns:
            Tuple of (response_text, metadata) if cache hit, None otherwise
        """
        cache_key = self._build_cache_key(query, items, mapping_keys, comparison_type, provider)
        
        # Exact match check first (fastest)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not self._is_expired(entry["timestamp"]):
                logger.info(f"✅ EXACT cache hit for key: {cache_key[:12]}...")
                return entry["response"], entry["metadata"]
            else:
                logger.debug(f"Cache entry expired for key: {cache_key[:12]}...")
                del self.cache[cache_key]
        
        # Semantic similarity search (slower but more flexible)
        query_embedding = self._compute_embedding(query)
        if query_embedding.size == 0:
            return None
        
        best_similarity = 0.0
        best_entry = None
        best_key = None
        
        for key, entry in list(self.cache.items()):
            # Skip expired entries
            if self._is_expired(entry["timestamp"]):
                del self.cache[key]
                continue
            
            # Only compare entries with same items, mapping keys, and provider
            if (
                entry["metadata"]["items"] != sorted([str(i).lower() for i in items]) or
                entry["metadata"]["mapping_keys"] != sorted(mapping_keys) or
                entry["metadata"]["provider"] != provider.lower()
            ):
                continue
            
            # Compute semantic similarity
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
                best_key = key
        
        # Return if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold and best_entry:
            logger.info(
                f"✅ SEMANTIC cache hit (similarity={best_similarity:.3f}) for key: {best_key[:12]}..."
            )
            return best_entry["response"], best_entry["metadata"]
        
        logger.debug(
            f"❌ Cache miss. Best similarity: {best_similarity:.3f} "
            f"(threshold: {self.similarity_threshold})"
        )
        return None
    
    def set(
        self,
        query: str,
        items: list,
        mapping_keys: list,
        comparison_type: str,
        provider: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store response in cache with semantic embedding.
        
        Args:
            query: User query
            items: Selected items (locations/cities/projects)
            mapping_keys: Selected mapping keys
            comparison_type: Type of comparison
            provider: LLM provider name
            response: Generated response text
            metadata: Optional additional metadata
        """
        cache_key = self._build_cache_key(query, items, mapping_keys, comparison_type, provider)
        query_embedding = self._compute_embedding(query)
        
        if query_embedding.size == 0:
            logger.warning("Failed to compute embedding, skipping cache storage")
            return
        
        entry_metadata = {
            "query": query,
            "items": sorted([str(i).lower() for i in items]),
            "mapping_keys": sorted(mapping_keys),
            "comparison_type": comparison_type.lower(),
            "provider": provider.lower(),
            **(metadata or {})
        }
        
        self.cache[cache_key] = {
            "embedding": query_embedding,
            "response": response,
            "metadata": entry_metadata,
            "timestamp": time.time()
        }
        
        logger.info(f"💾 Cached response for key: {cache_key[:12]}...")
        self._save_cache()
    
    def clear_expired(self):
        """Remove all expired entries from cache."""
        initial_count = len(self.cache)
        self.cache = {
            k: v for k, v in self.cache.items()
            if not self._is_expired(v["timestamp"])
        }
        removed = initial_count - len(self.cache)
        if removed > 0:
            logger.info(f"Cleared {removed} expired cache entries")
            self._save_cache()
    
    def clear_all(self):
        """Clear entire cache."""
        self.cache = {}
        self._save_cache()
        logger.info("Cleared all cache entries")

    def delete(
        self,
        query: str,
        items: list,
        mapping_keys: list,
        comparison_type: str,
        provider: str
    ) -> bool:
        """
        Delete a specific cache entry.
        Returns True if entry was found and deleted, False otherwise.
        """
        cache_key = self._build_cache_key(query, items, mapping_keys, comparison_type, provider)
        if cache_key in self.cache:
            del self.cache[cache_key]
            self._save_cache()
            logger.info(f"🗑️ Deleted cache entry for key: {cache_key[:12]}...")
            return True
        return False

    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = len(self.cache)
        expired = sum(1 for v in self.cache.values() if self._is_expired(v["timestamp"]))
        
        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "cache_file": str(self.cache_file),
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }
