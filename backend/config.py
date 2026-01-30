"""
Backend configuration for Strategy Supermarket.
MySQL database settings and application configuration.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""

    # MySQL Database Settings
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "strategy_market")

    # SQLAlchemy Database URL
    @property
    def SQLALCHEMY_DATABASE_URI(self):
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}?charset=utf8mb4"

    # Database Pool Settings
    SQLALCHEMY_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    SQLALCHEMY_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    SQLALCHEMY_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    SQLALCHEMY_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"

    # AkShare Rate Limiting (requests per second)
    AKSHARE_RATE_LIMIT = float(os.getenv("AKSHARE_RATE_LIMIT", "1.0"))
    AKSHARE_BURST_SIZE = int(os.getenv("AKSHARE_BURST_SIZE", "10"))

    # Data Settings
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
    DEFAULT_START_DATE = "20230101"  # Default backtest start date

    # Strategy Settings
    CSI300_STOCK_POOL = True  # Use CSI 300 as default stock pool
    MAX_POSITIONS_PER_STRATEGY = int(os.getenv("MAX_POSITIONS", "10"))

    # Scheduler Settings
    SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
    DATA_UPDATE_TIME = os.getenv("DATA_UPDATE_TIME", "15:30")  # Daily data update time

    # Redis Cache Settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.MYSQL_PASSWORD:
            print("Warning: MYSQL_PASSWORD not set, using empty password")
        return True


# Global config instance
config = Config()
config.validate()
