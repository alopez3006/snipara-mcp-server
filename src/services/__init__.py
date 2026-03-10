"""Services module for RLM MCP Server."""

from src.services.cache import QueryCache, get_cache
from src.services.chunker import Chunk, DocumentChunker, get_chunker
from src.services.embeddings import EmbeddingsService, get_embeddings_service
from src.services.indexer import DocumentIndexer, get_indexer
from src.services.tier_manager import (
    TIER_CONFIG,
    compute_tier,
    should_promote,
    update_chunk_access,
    batch_update_chunk_access,
)
from src.services.chunk_quality import (
    ChunkQuality,
    compute_chunk_quality,
    is_high_quality,
    quality_penalty,
)
from src.services.index_health import (
    IndexHealth,
    TierDistribution,
    QualityDistribution,
    StaleDocument,
    compute_index_health,
    get_index_recommendations,
)
from src.services.search_analytics import (
    SearchAnalytics,
    LatencyPercentiles,
    ToolUsage,
    DailyStats,
    compute_search_analytics,
    get_query_trends,
    get_top_queries,
)

__all__ = [
    "EmbeddingsService",
    "get_embeddings_service",
    "DocumentIndexer",
    "get_indexer",
    "DocumentChunker",
    "Chunk",
    "get_chunker",
    "QueryCache",
    "get_cache",
    # Tier management
    "TIER_CONFIG",
    "compute_tier",
    "should_promote",
    "update_chunk_access",
    "batch_update_chunk_access",
    # Chunk quality
    "ChunkQuality",
    "compute_chunk_quality",
    "is_high_quality",
    "quality_penalty",
    # Index health (Sprint 3)
    "IndexHealth",
    "TierDistribution",
    "QualityDistribution",
    "StaleDocument",
    "compute_index_health",
    "get_index_recommendations",
    # Search analytics (Sprint 3)
    "SearchAnalytics",
    "LatencyPercentiles",
    "ToolUsage",
    "DailyStats",
    "compute_search_analytics",
    "get_query_trends",
    "get_top_queries",
]
