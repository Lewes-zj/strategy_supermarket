# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strategy Supermarket (策略超市) is a quantitative trading strategy marketplace with a full-stack architecture:
- **Backend**: Python/FastAPI with a custom event-driven backtesting engine
- **Frontend**: React 19 + TypeScript + Vite

## Development Commands

### Backend (Python/FastAPI)
```bash
cd backend
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt
python main.py            # Runs on http://localhost:8000
```

### Frontend (React/TypeScript)
```bash
cd web
npm install
npm run dev               # Development server with hot reload
npm run build             # Production build (TypeScript check + Vite build)
npm run lint              # Run ESLint
npm run preview           # Preview production build
```

## Architecture

### Backend Structure
```
backend/
├── main.py                    # FastAPI app with CORS, in-memory cache
├── engine/
│   ├── backtester.py         # Core: Strategy, ExecutionModel, Portfolio, Order, Fill abstractions
│   └── data_loader.py        # Mock data generation (AkShare available but mock used for demo)
└── strategies/
    └── alpha_trend.py        # Example strategy implementation
```

**Key Patterns**:
- Strategies extend `Strategy` ABC with `on_bar()` and `on_fill()` methods
- `Backtester.run()` executes event loop: fills pending orders, updates equity, generates new orders
- Execution at T+1 Open (orders from T-1 execute at current bar prices)
- Results cached in-memory `STRATEGY_RESULTS` dict (no DB yet)

**API Endpoints**:
- `GET /api/strategies` - List all strategies with sparkline data
- `GET /api/strategies/{id}/metrics` - Performance metrics (CAGR, Sharpe, Max DD, Win Rate)
- `GET /api/strategies/{id}/equity_curve` - Equity over time with benchmark
- `GET /api/strategies/{id}/transactions` - Trade history with `is_subscribed` query param
- `GET /api/strategies/{id}/holdings` - Current positions, masks data if not subscribed

### Frontend Structure
```
web/src/
├── pages/
│   ├── Marketplace.tsx       # Strategy cards grid
│   ├── StrategyDetail.tsx    # Detail page with tabbed content
│   └── tabs/
│       ├── DataMetrics.tsx   # KPI cards + equity curve chart
│       ├── TransactionHistory.tsx  # Trade list with encryption for non-subscribers
│       └── DailyHoldings.tsx      # Holdings table with sector grouping
├── components/
│   ├── StrategyCard.tsx      # Card with metrics and sparkline
│   └── SubscriptionModal.tsx # Subscription flow
└── App.tsx                   # React Router setup
```

**Subscription Logic**: Non-subscribers see masked/encrypted data:
- Transactions: symbols replaced with sectors, prices hidden, fuzzy timestamps
- Holdings: symbols/names set to "HIDDEN", prices masked

## Adding a New Strategy

1. Create `backend/strategies/your_strategy.py`:
   ```python
   from engine.backtester import Strategy, Order, OrderSide, OrderType
   from datetime import datetime
   import pandas as pd

   class YourStrategy(Strategy):
       def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
           # Generate signals based on data up to timestamp
           return []

       def on_fill(self, fill: Fill) -> None:
           # Handle fill notifications
           pass
   ```

2. Register in `main.py`'s `get_strategy_result()` function
3. Frontend will automatically pick it up via `/api/strategies` endpoint

## Data Handling

- Mock data used for demo reliability (`generate_mock_data()` in `data_loader.py`)
- Real data available via AkShare (`fetch_stock_data()`) but commented out
- DataFrame index must be `pd.DatetimeIndex` for backtester
