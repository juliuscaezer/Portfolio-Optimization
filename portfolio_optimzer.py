import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, top_n=5):
        self.top_n = top_n

    def get_equal_weights(self, factor_scores):
        """
        Naive equal weighting (1/N) baseline for the top N ranked stocks.
        """
        final_rankings = factor_scores.rank(axis=1, method='min')
        monthly_ranks = final_rankings.resample('ME').last() # Fixed deprecation warning
        
        weights = pd.DataFrame(0.0, index=monthly_ranks.index, columns=monthly_ranks.columns)
        
        for date, row in monthly_ranks.iterrows():
            top_stocks = row.nsmallest(self.top_n).index.tolist()
            
            # Prevent division by zero if a month has no valid data
            if len(top_stocks) == 0:
                continue
                
            weights.loc[date, top_stocks] = 1.0 / len(top_stocks)
            
        return weights

    def get_mean_variance_weights(self, factor_scores, returns_df, lookback_window=63):
        """
        Uses Markowitz Mean-Variance Optimization to size positions.
        """
        final_rankings = factor_scores.rank(axis=1, method='min')
        monthly_ranks = final_rankings.resample('ME').last() # Fixed deprecation warning
        
        weights = pd.DataFrame(0.0, index=monthly_ranks.index, columns=monthly_ranks.columns)
        
        for date, row in monthly_ranks.iterrows():
            top_stocks = row.nsmallest(self.top_n).index.tolist()
            
            num_assets = len(top_stocks)
            # Catch the ZeroDivisionError: If no stocks are returned, skip this month
            if num_assets == 0:
                continue
            
            hist_returns = returns_df.loc[:date].tail(lookback_window)[top_stocks]
            
            # If we don't have enough rows of historical data, fallback to equal weights
            if len(hist_returns) < 20: 
                weights.loc[date, top_stocks] = 1.0 / num_assets
                continue
                
            cov_matrix = hist_returns.cov().values * 252
            mean_returns = hist_returns.mean().values * 252
            
            def neg_sharpe(w):
                port_return = np.dot(w, mean_returns)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                if port_vol == 0:
                    return 0
                return -(port_return / port_vol)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            bounds = tuple((0.0, 1.0) for _ in range(num_assets))
            
            # Initialize with equal weights
            init_guess = num_assets * [1.0 / num_assets]
            
            opt_result = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=constraints)
            
            if opt_result.success:
                opt_weights = opt_result.x
            else:
                opt_weights = init_guess
                
            weights.loc[date, top_stocks] = opt_weights
            
        return weights