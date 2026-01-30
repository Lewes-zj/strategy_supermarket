# services/rate_limiter.py
"""
Adaptive Rate Limiter - prevents AKShare API requests from being too fast and causing IP bans.
Uses Token Bucket algorithm with adaptive rate adjustment based on success/failure.
"""

import time
import threading
from typing import Optional


class AdaptiveRateLimiter:
    """
    Adaptive Rate Limiter using Token Bucket algorithm.

    Features:
    - Token bucket for smooth rate limiting with burst support
    - Adaptive rate: backs off on errors, speeds up on success
    - Thread-safe operation

    Args:
        base_rate: Base request rate (tokens per second). Default 1.0
        min_rate: Minimum rate when backing off. Default 0.2 (1 request per 5 seconds)
        max_rate: Maximum rate when speeding up. Default 2.0
        burst_size: Maximum tokens in bucket (burst capacity). Default 10
    """

    def __init__(
        self,
        base_rate: float = 1.0,      # Base rate: 1 request per second
        min_rate: float = 0.2,       # Min rate: 1 request per 5 seconds
        max_rate: float = 2.0,       # Max rate: 2 requests per second
        burst_size: int = 10         # Burst capacity
    ):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.consecutive_success = 0
        self.consecutive_fail = 0
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token to make a request.

        Args:
            blocking: If True, wait until a token is available. Default True.
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if token was acquired, False if timed out or non-blocking and no token available.
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            if not blocking:
                return False

            if timeout is not None and (time.time() - start_time) > timeout:
                return False

            # Sleep a short time before retrying
            time.sleep(0.1)

    def _refill(self):
        """Refill tokens based on elapsed time and current rate."""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.current_rate
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_update = now

    def on_success(self):
        """
        Callback when a request succeeds.

        Gradually increases rate after 10 consecutive successes,
        up to max_rate.
        """
        with self._lock:
            self.consecutive_success += 1
            self.consecutive_fail = 0
            if self.consecutive_success >= 10:
                self.current_rate = min(self.current_rate * 1.2, self.max_rate)
                self.consecutive_success = 0

    def on_fail(self):
        """
        Callback when a request fails.

        Immediately reduces rate by 50% (down to min_rate),
        and applies exponential backoff sleep.
        """
        with self._lock:
            self.consecutive_fail += 1
            self.consecutive_success = 0
            self.current_rate = max(self.current_rate * 0.5, self.min_rate)

            # Exponential backoff sleep (max 60 seconds)
            backoff = min(2 ** self.consecutive_fail, 60)

        # Sleep outside the lock to not block other threads
        time.sleep(backoff)

    def get_current_rate(self) -> float:
        """Get the current rate limit (tokens per second)."""
        with self._lock:
            return self.current_rate

    def reset(self):
        """Reset the rate limiter to initial state."""
        with self._lock:
            self.current_rate = self.base_rate
            self.tokens = float(self.burst_size)
            self.last_update = time.time()
            self.consecutive_success = 0
            self.consecutive_fail = 0

    def __enter__(self):
        """Context manager entry - acquires a token."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - calls on_success or on_fail based on exception."""
        if exc_type is None:
            self.on_success()
        else:
            self.on_fail()
        return False  # Don't suppress exceptions


# Global instance for AKShare requests
_adaptive_limiter: Optional[AdaptiveRateLimiter] = None


def get_adaptive_limiter() -> AdaptiveRateLimiter:
    """Get or create the global adaptive rate limiter instance."""
    global _adaptive_limiter
    if _adaptive_limiter is None:
        _adaptive_limiter = AdaptiveRateLimiter()
    return _adaptive_limiter
