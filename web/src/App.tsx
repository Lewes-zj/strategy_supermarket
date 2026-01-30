import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Marketplace from './pages/Marketplace';
import StrategyDetail from './pages/StrategyDetail';
import DataMetrics from './pages/tabs/DataMetrics';
import TransactionHistory from './pages/tabs/TransactionHistory';
import DailyHoldings from './pages/tabs/DailyHoldings';

function App() {
  return (
    <Router>
      <div className="app-container">
        {/* Global Nav could go here */}
        <Routes>
          <Route path="/" element={<Marketplace />} />
          <Route path="/strategy/:id" element={<StrategyDetail />}>
            <Route index element={<Navigate to="metrics" replace />} />
            <Route path="metrics" element={<DataMetrics />} />
            <Route path="transactions" element={<TransactionHistory />} />
            <Route path="holdings" element={<DailyHoldings />} />
          </Route>
        </Routes>
      </div>
    </Router>
  );
}

export default App;
