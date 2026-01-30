"""
Redis cache service for Strategy Supermarket.
Provides caching layer for strategy data with year-based segmentation.
"""
import json
import logging
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import redis, gracefully handle if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Running without cache.")


class CacheService:
    """Redis cache service with graceful degradation."""

    def __init__(self):
        self._client: Optional[Any] = None
        self._connected = False

    def connect(self, host: str = "localhost", port: int = 6379,
                password: str = "", db: int = 0) -> bool:
        """
        Connect to Redis server.
        Returns True if connected, False otherwise.
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not installed")
            return False

        try:
            self._client = redis.Redis(
                host=host,
                port=port,
                password=password or None,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Redis connected: {host}:{port}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running without cache.")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._client is not None

    def _make_key(self, strategy_id: str, data_type: str, year: Optional[int] = None) -> str:
        """
        Generate cache key.
        Format: strategy:{id}:{type}:{year} or strategy:{id}:{type}:all
        """
        year_part = str(year) if year else "all"
        return f"strategy:{strategy_id}:{data_type}:{year_part}"

    def _get_ttl(self, year: Optional[int]) -> int:
        """
        Get TTL based on year.
        - Historical years: 24 hours (data won't change)
        - Current year: 1 hour (may have new trades)
        - All (no year): 1 hour
        """
        if year is None:
            return 3600  # 1 hour for full period

        current_year = datetime.now().year
        if year < current_year:
            return 86400  # 24 hours for historical
        else:
            return 3600  # 1 hour for current year

    def get(self, strategy_id: str, data_type: str, year: Optional[int] = None) -> Optional[Any]:
        """
        Get cached data.
        Returns None if not cached or Redis unavailable.
        """
        if not self.is_connected:
            return None

        try:
            key = self._make_key(strategy_id, data_type, year)
            data = self._client.get(key)
            if data:
                logger.debug(f"Cache hit: {key}")
                return json.loads(data)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, strategy_id: str, data_type: str, data: Any,
            year: Optional[int] = None) -> bool:
        """
        Set cached data with appropriate TTL.
        Returns True if successful.
        """
        if not self.is_connected:
            return False

        try:
            key = self._make_key(strategy_id, data_type, year)
            ttl = self._get_ttl(year)
            self._client.setex(key, ttl, json.dumps(data))
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    def invalidate(self, strategy_id: str, data_type: Optional[str] = None) -> int:
        """
        Invalidate cached data for a strategy.
        If data_type is None, invalidates all data types.
        Returns number of keys deleted.
        """
        if not self.is_connected:
            return 0

        try:
            if data_type:
                pattern = f"strategy:{strategy_id}:{data_type}:*"
            else:
                pattern = f"strategy:{strategy_id}:*"

            keys = list(self._client.scan_iter(match=pattern))
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Cache invalidated: {pattern} ({deleted} keys)")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidate failed: {e}")
            return 0

    def invalidate_all(self) -> int:
        """
        Invalidate all strategy cache.
        Use with caution.
        """
        if not self.is_connected:
            return 0

        try:
            pattern = "strategy:*"
            keys = list(self._client.scan_iter(match=pattern))
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"All cache invalidated ({deleted} keys)")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidate all failed: {e}")
            return 0


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get or create the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        # Try to connect using config
        try:
            from config import config
            _cache_service.connect(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB
            )
        except Exception as e:
            logger.warning(f"Failed to initialize cache from config: {e}")
            # Try default connection
            _cache_service.connect()
    return _cache_service
