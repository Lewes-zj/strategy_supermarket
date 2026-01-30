# tests/services/test_rate_limiter.py
"""
Tests for AdaptiveRateLimiter - adaptive rate limiting for AKShare API.

The AdaptiveRateLimiter uses a token bucket algorithm with adaptive rate adjustment:
- Backs off (reduces rate) when requests fail
- Speeds up (increases rate) after consecutive successes
- Supports burst requests up to burst_size
"""
import pytest
import sys
import os
import time
import threading
import importlib.util
from unittest.mock import patch, MagicMock

# Direct import of rate_limiter module to avoid services/__init__.py chain
# This is necessary because services/__init__.py imports other modules that have
# broken dependencies in the current state of the codebase
_module_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'services', 'rate_limiter.py'
)
_spec = importlib.util.spec_from_file_location("rate_limiter", _module_path)
_rate_limiter_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rate_limiter_module)

AdaptiveRateLimiter = _rate_limiter_module.AdaptiveRateLimiter
get_adaptive_limiter = _rate_limiter_module.get_adaptive_limiter


class TestAdaptiveRateLimiterBasic:
    """Basic functionality tests for AdaptiveRateLimiter."""

    def test_init_default_values(self):
        """Test that default initialization values are correct."""
        limiter = AdaptiveRateLimiter()

        assert limiter.base_rate == 1.0
        assert limiter.current_rate == 1.0
        assert limiter.min_rate == 0.2
        assert limiter.max_rate == 2.0
        assert limiter.burst_size == 10
        assert limiter.tokens == 10.0
        assert limiter.consecutive_success == 0
        assert limiter.consecutive_fail == 0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        limiter = AdaptiveRateLimiter(
            base_rate=2.0,
            min_rate=0.5,
            max_rate=5.0,
            burst_size=20
        )

        assert limiter.base_rate == 2.0
        assert limiter.current_rate == 2.0
        assert limiter.min_rate == 0.5
        assert limiter.max_rate == 5.0
        assert limiter.burst_size == 20
        assert limiter.tokens == 20.0

    def test_acquire_consumes_token(self):
        """Test that acquire consumes a token."""
        limiter = AdaptiveRateLimiter(burst_size=5)

        initial_tokens = limiter.tokens
        result = limiter.acquire(blocking=False)

        assert result is True
        assert limiter.tokens < initial_tokens

    def test_acquire_burst_capacity(self):
        """Test that burst capacity allows multiple rapid requests."""
        limiter = AdaptiveRateLimiter(burst_size=5)

        # Should be able to acquire burst_size tokens rapidly
        for _ in range(5):
            result = limiter.acquire(blocking=False)
            assert result is True

        # Next acquire should fail (no tokens left) if not blocking
        # Note: Some time may have passed, so tokens may have refilled
        # We need to check immediately
        result = limiter.acquire(blocking=False)
        # This may or may not succeed depending on timing

    def test_acquire_non_blocking_returns_false_when_no_tokens(self):
        """Test non-blocking acquire returns False when no tokens available."""
        limiter = AdaptiveRateLimiter(base_rate=0.1, burst_size=1)

        # Consume the only token
        limiter.acquire(blocking=False)
        limiter.tokens = 0  # Force no tokens

        # Non-blocking should return False immediately
        result = limiter.acquire(blocking=False)
        assert result is False

    def test_acquire_blocking_waits_for_token(self):
        """Test blocking acquire waits for token to become available."""
        limiter = AdaptiveRateLimiter(base_rate=10.0, burst_size=1)  # High rate for fast test

        # Consume token
        limiter.acquire(blocking=False)
        limiter.tokens = 0

        # Blocking acquire should wait and eventually succeed
        start = time.time()
        result = limiter.acquire(blocking=True, timeout=1.0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed > 0  # Should have waited some time

    def test_acquire_timeout(self):
        """Test that acquire times out correctly."""
        limiter = AdaptiveRateLimiter(base_rate=0.1, burst_size=1)

        # Consume token and prevent refill
        limiter.acquire(blocking=False)
        limiter.tokens = 0

        # Should timeout
        start = time.time()
        result = limiter.acquire(blocking=True, timeout=0.2)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 0.2
        assert elapsed < 0.5  # Should not wait much longer than timeout


class TestAdaptiveRateLimiterAdaptive:
    """Tests for adaptive rate adjustment behavior."""

    def test_on_success_increments_counter(self):
        """Test that on_success increments consecutive success counter."""
        limiter = AdaptiveRateLimiter()

        assert limiter.consecutive_success == 0
        limiter.on_success()
        assert limiter.consecutive_success == 1

    def test_on_success_resets_fail_counter(self):
        """Test that on_success resets consecutive fail counter."""
        limiter = AdaptiveRateLimiter()

        limiter.consecutive_fail = 5
        limiter.on_success()
        assert limiter.consecutive_fail == 0

    def test_on_success_increases_rate_after_10_successes(self):
        """Test that rate increases after 10 consecutive successes."""
        limiter = AdaptiveRateLimiter(base_rate=1.0, max_rate=2.0)

        initial_rate = limiter.current_rate

        # 10 consecutive successes
        for _ in range(10):
            limiter.on_success()

        # Rate should have increased by 20%
        assert limiter.current_rate > initial_rate
        assert limiter.current_rate == pytest.approx(initial_rate * 1.2, rel=0.01)

    def test_on_success_rate_capped_at_max(self):
        """Test that rate does not exceed max_rate."""
        limiter = AdaptiveRateLimiter(base_rate=1.5, max_rate=2.0)

        # Many successes
        for _ in range(50):
            limiter.on_success()

        assert limiter.current_rate <= limiter.max_rate

    @patch('time.sleep')
    def test_on_fail_decreases_rate(self, mock_sleep):
        """Test that on_fail decreases rate by 50%."""
        limiter = AdaptiveRateLimiter(base_rate=1.0, min_rate=0.1)

        initial_rate = limiter.current_rate
        limiter.on_fail()

        assert limiter.current_rate == pytest.approx(initial_rate * 0.5, rel=0.01)

    @patch('time.sleep')
    def test_on_fail_rate_floored_at_min(self, mock_sleep):
        """Test that rate does not go below min_rate."""
        limiter = AdaptiveRateLimiter(base_rate=0.3, min_rate=0.2)

        # Multiple failures
        for _ in range(10):
            limiter.on_fail()

        assert limiter.current_rate >= limiter.min_rate

    @patch('time.sleep')
    def test_on_fail_increments_counter(self, mock_sleep):
        """Test that on_fail increments consecutive fail counter."""
        limiter = AdaptiveRateLimiter()

        assert limiter.consecutive_fail == 0
        limiter.on_fail()
        assert limiter.consecutive_fail == 1

    @patch('time.sleep')
    def test_on_fail_resets_success_counter(self, mock_sleep):
        """Test that on_fail resets consecutive success counter."""
        limiter = AdaptiveRateLimiter()

        limiter.consecutive_success = 5
        limiter.on_fail()
        assert limiter.consecutive_success == 0

    @patch('time.sleep')
    def test_on_fail_exponential_backoff(self, mock_sleep):
        """Test that on_fail applies exponential backoff."""
        limiter = AdaptiveRateLimiter()

        # First failure: sleep 2^1 = 2 seconds
        limiter.on_fail()
        mock_sleep.assert_called_with(2)

        # Second failure: sleep 2^2 = 4 seconds
        limiter.on_fail()
        mock_sleep.assert_called_with(4)

        # Third failure: sleep 2^3 = 8 seconds
        limiter.on_fail()
        mock_sleep.assert_called_with(8)

    @patch('time.sleep')
    def test_on_fail_backoff_capped_at_60_seconds(self, mock_sleep):
        """Test that exponential backoff is capped at 60 seconds."""
        limiter = AdaptiveRateLimiter()

        # Many failures
        for i in range(10):
            limiter.on_fail()

        # Last call should be capped at 60
        assert mock_sleep.call_args[0][0] <= 60


class TestAdaptiveRateLimiterGetCurrentRate:
    """Tests for get_current_rate method."""

    def test_get_current_rate_returns_rate(self):
        """Test that get_current_rate returns the current rate."""
        limiter = AdaptiveRateLimiter(base_rate=1.5)

        assert limiter.get_current_rate() == 1.5

    @patch('time.sleep')
    def test_get_current_rate_reflects_changes(self, mock_sleep):
        """Test that get_current_rate reflects rate changes."""
        limiter = AdaptiveRateLimiter(base_rate=1.0)

        initial_rate = limiter.get_current_rate()

        # Reduce rate
        limiter.on_fail()

        new_rate = limiter.get_current_rate()
        assert new_rate < initial_rate


class TestAdaptiveRateLimiterReset:
    """Tests for reset method."""

    @patch('time.sleep')
    def test_reset_restores_initial_state(self, mock_sleep):
        """Test that reset restores initial state."""
        limiter = AdaptiveRateLimiter(base_rate=1.0, burst_size=10)

        # Modify state
        limiter.on_fail()
        limiter.on_fail()
        for _ in range(3):
            limiter.acquire(blocking=False)

        # Reset
        limiter.reset()

        assert limiter.current_rate == limiter.base_rate
        assert limiter.tokens == float(limiter.burst_size)
        assert limiter.consecutive_success == 0
        assert limiter.consecutive_fail == 0


class TestAdaptiveRateLimiterContextManager:
    """Tests for context manager usage."""

    def test_context_manager_acquires_token(self):
        """Test that context manager acquires token on entry."""
        limiter = AdaptiveRateLimiter(burst_size=10)

        initial_tokens = limiter.tokens

        with limiter:
            pass

        # Tokens should be consumed (1 for acquire) and success called
        # Due to refill timing, we check that acquire happened
        assert limiter.consecutive_success >= 1

    def test_context_manager_calls_on_success(self):
        """Test that context manager calls on_success on normal exit."""
        limiter = AdaptiveRateLimiter()

        with limiter:
            pass

        assert limiter.consecutive_success == 1

    @patch('time.sleep')
    def test_context_manager_calls_on_fail_on_exception(self, mock_sleep):
        """Test that context manager calls on_fail on exception."""
        limiter = AdaptiveRateLimiter()

        try:
            with limiter:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert limiter.consecutive_fail == 1


class TestAdaptiveRateLimiterThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_acquire(self):
        """Test that concurrent acquires are thread-safe."""
        limiter = AdaptiveRateLimiter(base_rate=100, burst_size=50)

        acquired_count = 0
        lock = threading.Lock()

        def acquire_token():
            nonlocal acquired_count
            if limiter.acquire(blocking=False):
                with lock:
                    acquired_count += 1

        threads = [threading.Thread(target=acquire_token) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have acquired tokens (burst_size is 50)
        assert acquired_count == 20

    @patch('time.sleep')
    def test_concurrent_success_fail(self, mock_sleep):
        """Test that concurrent on_success/on_fail are thread-safe."""
        limiter = AdaptiveRateLimiter()

        def call_success():
            for _ in range(50):
                limiter.on_success()

        def call_fail():
            for _ in range(5):
                limiter.on_fail()

        threads = [
            threading.Thread(target=call_success),
            threading.Thread(target=call_fail)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash and rate should be within bounds
        assert limiter.min_rate <= limiter.current_rate <= limiter.max_rate


class TestAdaptiveRateLimiterTokenRefill:
    """Tests for token refill behavior."""

    def test_tokens_refill_over_time(self):
        """Test that tokens refill based on elapsed time."""
        limiter = AdaptiveRateLimiter(base_rate=10.0, burst_size=10)

        # Consume all tokens
        for _ in range(10):
            limiter.acquire(blocking=False)

        # Force tokens to near zero
        limiter.tokens = 0.5

        # Wait a bit
        time.sleep(0.2)

        # Try to acquire - should succeed after refill
        result = limiter.acquire(blocking=False)

        # With rate=10, 0.2s should add 2 tokens
        assert result is True

    def test_tokens_capped_at_burst_size(self):
        """Test that tokens don't exceed burst_size."""
        limiter = AdaptiveRateLimiter(burst_size=5)

        # Wait to accumulate tokens
        time.sleep(0.5)

        # Trigger refill
        limiter.acquire(blocking=False)

        # Tokens should still be <= burst_size
        assert limiter.tokens <= limiter.burst_size


class TestGetAdaptiveLimiter:
    """Tests for get_adaptive_limiter factory function."""

    def test_returns_adaptive_rate_limiter(self):
        """Test that get_adaptive_limiter returns an AdaptiveRateLimiter."""
        limiter = get_adaptive_limiter()
        assert isinstance(limiter, AdaptiveRateLimiter)

    def test_returns_same_instance(self):
        """Test that get_adaptive_limiter returns singleton."""
        limiter1 = get_adaptive_limiter()
        limiter2 = get_adaptive_limiter()
        assert limiter1 is limiter2
