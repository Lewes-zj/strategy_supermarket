"""
Rate limiter using Token Bucket algorithm.
Used for controlling AkShare API request rate to avoid IP blocking.
"""
import time
import threading
from typing import Callable
from functools import wraps
import logging

from config import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token Bucket rate limiter.

    Allows bursts of requests up to burst_size, then limits to rate requests/second.
    """

    def __init__(self, rate: float, burst_size: int):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens added per second
            burst_size: Maximum bucket size (max burst allowed)
        """
        self.rate = rate
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        Acquire a token.

        Args:
            blocking: If True, block until token is available
            timeout: Maximum time to wait (None = infinite)

        Returns:
            True if token acquired, False otherwise
        """
        with self._lock:
            self._add_tokens()

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            if not blocking:
                return False

            # Calculate wait time
            tokens_needed = 1 - self.tokens
            wait_time = tokens_needed / self.rate

            if timeout is not None and wait_time > timeout:
                return False

        # Wait outside lock
        time.sleep(wait_time)

        # Retry acquisition
        with self._lock:
            self._add_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


# Global rate limiter instance
_akshare_limiter = None


def get_akshare_limiter() -> RateLimiter:
    """Get the global AkShare rate limiter instance."""
    global _akshare_limiter

    if _akshare_limiter is None:
        _akshare_limiter = RateLimiter(
            rate=config.AKSHARE_RATE_LIMIT,
            burst_size=config.AKSHARE_BURST_SIZE
        )
        logger.info(f"Created AkShare rate limiter: rate={config.AKSHARE_RATE_LIMIT}/s, burst={config.AKSHARE_BURST_SIZE}")

    return _akshare_limiter


def rate_limit(rps: float = None, burst: int = None):
    """
    Decorator to rate limit a function.

    Args:
        rps: Requests per second (default from config)
        burst: Burst size (default from config)

    Usage:
        @rate_limit(rps=1.0, burst=10)
        def fetch_data():
            ...
    """
    if rps is None:
        rps = config.AKSHARE_RATE_LIMIT
    if burst is None:
        burst = config.AKSHARE_BURST_SIZE

    limiter = RateLimiter(rps, burst)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire(blocking=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator
