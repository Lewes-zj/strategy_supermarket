"""
Monkey patch AkShare to add browser-like headers for bypassing anti-scraping protection.

This module patches the requests session used by AkShare to simulate a real browser.
"""
import logging
import random

logger = logging.getLogger(__name__)

# Realistic browser User-Aengers
USER_AGENTS = [
    # Chrome on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    # Chrome on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    # Safari on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    # Firefox
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
    # Edge
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
]


def get_random_user_agent() -> str:
    """Get a random User-Agent string."""
    return random.choice(USER_AGENTS)


def patch_requests_session():
    """
    Patch requests.Session to add browser-like headers.

    This modifies the session's request method to inject headers before each request.
    """
    import requests
    from urllib.parse import urlparse

    # Store the original request method
    original_request = requests.Session.request

    def patched_request(self, method, url, *args, **kwargs):
        """Patched request method with browser-like headers."""
        # Parse URL to check domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Add browser headers for eastmoney.com and similar domains
        if any(d in domain for d in ['eastmoney.com', 'push2his.eastmoney.com', 'akshare.akfamily.xyz']):
            # Get headers from kwargs or create new dict
            headers = kwargs.get('headers', {})

            # Browser-like headers
            browser_headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
                'DNT': '1',
            }

            # Merge headers (kwargs headers take precedence)
            for key, value in browser_headers.items():
                if key not in headers:
                    headers[key] = value

            kwargs['headers'] = headers

            # Disable proxy for these domains (use NO_PROXY)
            kwargs['proxies'] = {
                'http': None,
                'https': None,
            }

            # Set timeout
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 30

        # Call original request method
        return original_request(self, method, url, *args, **kwargs)

    # Apply the patch
    requests.Session.request = patched_request
    logger.info("Patched requests.Session with browser-like headers")


def patch_akshare():
    """
    Patch AkShare module to use browser-like headers.

    This should be called before any AkShare functions are used.
    """
    try:
        # First patch requests.Session
        patch_requests_session()

        # Also try to patch akshare's internal session if it exists
        import akshare as ak

        # Check if akshare has a session module
        if hasattr(ak, 'stock_zh_a_hist'):
            # Get the function
            func = ak.stock_zh_a_hist

            # Store original function
            original_func = func

            def wrapped_stock_zh_a_hist(*args, **kwargs):
                """Wrapped function with custom headers."""
                # Add retry logic
                import time
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return original_func(*args, **kwargs)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
                            logger.warning(f"Request failed, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            raise

            # Don't wrap here - requests.Session patch is enough
            pass

        logger.info("AkShare patched successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to patch AkShare: {e}")
        return False


# Auto-patch on import
_patch_done = False


def ensure_patched():
    """Ensure AkShare is patched (call this before using AkShare)."""
    global _patch_done
    if not _patch_done:
        patch_akshare()
        _patch_done = True
    return _patch_done


# Patch immediately when module is imported
ensure_patched()
