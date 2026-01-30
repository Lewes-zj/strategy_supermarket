"""
Initialize stock pool with CSI 300 constituents.
Run this script to populate the stock pool database.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database.connection import init_db
from services.data_service import get_data_service


def main():
    """Initialize stock pool with CSI 300 data."""
    print("=" * 60)
    print("Strategy Supermarket - Stock Pool Initialization")
    print("=" * 60)

    # Initialize database tables
    print("\n1. Initializing database tables...")
    try:
        init_db(drop_tables=False)
        print("   ✓ Database tables ready")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
        print("   Please check your MySQL connection in .env file")
        return

    # Initialize stock pool
    print("\n2. Fetching CSI 300 stock pool from AkShare...")
    try:
        data_service = get_data_service()
        count = data_service.init_stock_pool()
        print(f"   ✓ Added {count} symbols to stock pool")
    except Exception as e:
        print(f"   ✗ Stock pool initialization failed: {e}")
        return

    # Update historical data (last 2 years)
    print("\n3. Fetching historical data (this may take a while)...")
    try:
        stats = data_service.update_stock_data(days_back=365*2)
        print(f"   ✓ Updated {stats['updated']} records")
        print(f"   - Skipped: {stats['skipped']} (already up to date)")
        if stats['failed'] > 0:
            print(f"   - Failed: {stats['failed']}")
    except Exception as e:
        print(f"   ✗ Data update failed: {e}")
        return

    print("\n" + "=" * 60)
    print("Initialization complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the backend server: python main.py")
    print("2. Start the frontend server: cd web && npm run dev")
    print("3. Visit http://localhost:5173/")


if __name__ == "__main__":
    main()
