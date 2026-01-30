"""
Database connection management for Strategy Supermarket.
Uses SQLAlchemy with MySQL connection pooling.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging

from config import config
from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session maker
_engine = None
_session_local = None


def get_engine():
    """Get or create the database engine with connection pooling."""
    global _engine

    if _engine is None:
        # Create engine with connection pooling
        _engine = create_engine(
            config.SQLALCHEMY_DATABASE_URI,
            poolclass=QueuePool,
            pool_size=config.SQLALCHEMY_POOL_SIZE,
            max_overflow=config.SQLALCHEMY_MAX_OVERFLOW,
            pool_recycle=config.SQLALCHEMY_POOL_RECYCLE,
            pool_pre_ping=True,  # Verify connections before using
            echo=config.SQLALCHEMY_ECHO,
        )

        # Add connection event handlers for better logging
        @event.listens_for(_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("Database connection established")

        @event.listens_for(_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

        logger.info(f"Database engine created: {config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}")

    return _engine


def get_session_local():
    """Get or create the session factory."""
    global _session_local

    if _session_local is None:
        engine = get_engine()
        _session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

    return _session_local


@contextmanager
def get_session():
    """
    Context manager for database sessions.
    Automatically handles commit/rollback and cleanup.

    Usage:
        with get_session() as session:
            session.query(...)
    """
    session_local = get_session_local()
    session = session_local()

    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def init_db(drop_tables=False):
    """
    Initialize the database schema.

    Args:
        drop_tables: If True, drop all existing tables first (use with caution!)
    """
    engine = get_engine()

    if drop_tables:
        logger.warning("Dropping all existing tables!")
        Base.metadata.drop_all(bind=engine)

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")


def test_connection():
    """Test database connection."""
    try:
        with get_session() as session:
            session.execute("SELECT 1")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
