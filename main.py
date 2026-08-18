import matplotlib.pyplot as plt
import pandas as pd
from data_loader import DataLoader
from alpha_factors import AlphaFactors
from portfolio_optimzer import PortfolioOptimizer
from backtest_engine import BacktestEngine
from metrics import get_performance_summary

# 1. Fetch Data
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'META', 'BRK-B', 'TSLA', 'UNH', 'JNJ',
           'V', 'PG', 'MA', 'XOM', 'AVGO', 'LLY', 'HD', 'JPM', 'CVX', 'MRK']

loader = DataLoader(tickers, start_date="2010-01-01", end_date="2025-01-01")
close, volume = loader.get_clean_data()

# 2. Calculate Alpha Factors
alpha_model = AlphaFactors(close, volume, window=21)
factor_scores = alpha_model.get_factor_scores()

# 3. Generate Weights
optimizer = PortfolioOptimizer(top_n=5)
returns = close.pct_change().fillna(0)

# Baseline: 1/N Equal Weighting
baseline_weights = optimizer.get_equal_weights(factor_scores)

# Optimized: Mean-Variance
optimized_weights = optimizer.get_mean_variance_weights(factor_scores, returns, lookback_window=63)

# 4. Backtest Both Strategies
engine = BacktestEngine(close, transaction_cost=0.0005)

print("\n--- Running Baseline (1/N) Backtest ---")
base_net_returns, base_cum_returns = engine.run(baseline_weights)

print("\n--- Running Optimized (Mean-Variance) Backtest ---")
opt_net_returns, opt_cum_returns = engine.run(optimized_weights)

# 5. Print Performance Metrics
print("\n[Baseline 1/N Metrics]")
for k, v in get_performance_summary(base_net_returns).items():
    print(f"{k}: {v:.4f}")

print("\n[Optimized Mean-Variance Metrics]")
for k, v in get_performance_summary(opt_net_returns).items():
    print(f"{k}: {v:.4f}")

# 6. Plot the Comparison
plt.figure(figsize=(12, 6))
plt.plot(base_cum_returns, label="Baseline (1/N)", linestyle="--", alpha=0.8)
plt.plot(opt_cum_returns, label="Optimized (Mean-Variance)", linewidth=2)
plt.title("Factor Strategy: Mean-Variance vs Baseline (1/N)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()