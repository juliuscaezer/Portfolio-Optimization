import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Calculates the annualized Sharpe Ratio."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())

def calculate_max_drawdown(returns):
    """Calculates the Maximum Drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def get_performance_summary(returns):
    """Returns a dictionary of performance metrics."""
    return {
        "Annualized Return": (1 + returns.mean()) ** 252 - 1,
        "Annualized Volatility": returns.std() * np.sqrt(252),
        "Sharpe Ratio": calculate_sharpe_ratio(returns),
        "Max Drawdown": calculate_max_drawdown(returns)
    }