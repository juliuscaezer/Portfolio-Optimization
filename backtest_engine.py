import pandas as pd

class BacktestEngine:
    def __init__(self, close_prices, transaction_cost=0.0005):
        """
        Initializes the backtest engine.
        :param close_prices: DataFrame of daily closing prices.
        :param transaction_cost: Proportional cost per trade (e.g., 0.0005 = 5 bps).
        """
        self.close = close_prices
        self.returns = close_prices.pct_change().fillna(0)
        self.transaction_cost = transaction_cost

    def run(self, target_weights):
        """
        Runs the backtest using the target weights.
        Ensures strict temporal alignment to avoid look-ahead bias.
        """
        # Safely align calendar month-end weights to actual trading days
        # Combine both indexes so we don't lose the weight generation dates
        combined_index = self.close.index.union(target_weights.index).sort_values()
        
        # Reindex to the combined index, forward fill the weights, then slice out only the trading days
        daily_weights = target_weights.reindex(combined_index).ffill().reindex(self.close.index)
        
        # **Crucial Fix**: Shift weights by 1 day! 
        # Signals generated at the close of T are traded at the close of T, so they earn T+1's return.
        shifted_weights = daily_weights.shift(1).fillna(0)
        
        # Element-wise multiply shifted_weights by today's returns
        portfolio_returns = (shifted_weights * self.returns).sum(axis=1)
        
        # Calculate daily turnover to apply transaction costs
        weight_changes = shifted_weights.diff().fillna(0)
        turnover = weight_changes.abs().sum(axis=1) / 2.0  # divided by 2 to avoid double counting buys and sells
        
        costs = turnover * self.transaction_cost
        
        # Net returns after slippage/commissions
        net_returns = portfolio_returns - costs
        
        cumulative_returns = (1 + net_returns).cumprod()
        
        return net_returns, cumulative_returns