import pandas as pd

class AlphaFactors:
    def __init__(self, close_prices, volume, window=21):
        self.close = close_prices
        self.volume = volume
        self.window = window

    def compute_momentum(self):
        """Calculates momentum as the percentage change over the window."""
        return self.close.pct_change(periods=self.window)

    def compute_volatility(self):
        """Calculates volatility as the rolling standard deviation of daily returns."""
        returns = self.close.pct_change()
        return returns.rolling(window=self.window).std()

    def compute_size(self):
        """Calculates size as the rolling average of Close * Volume."""
        size = self.close * self.volume
        return size.rolling(window=self.window).mean()

    def get_factor_scores(self):
        """
        Calculates composite factor scores by ranking cross-sectionally.
        Smaller rank sum is better.
        - Momentum: Higher is better -> ascending=False
        - Volatility: Lower is better -> ascending=True (Fixed from naive implementation)
        - Size: Higher is better -> ascending=False
        """
        momentum = self.compute_momentum()
        volatility = self.compute_volatility()
        size = self.compute_size()

        # Get valid index intersection where all factors are available
        valid_index = momentum.dropna().index \
            .intersection(volatility.dropna().index) \
            .intersection(size.dropna().index)

        momentum = momentum.loc[valid_index]
        volatility = volatility.loc[valid_index]
        size = size.loc[valid_index]

        # Rank cross-sectionally daily
        def rank(df, ascending=True):
            return df.rank(axis=1, method='first', ascending=ascending)

        # Sum of ranks (smaller sum means it ranked highly across all 3 factors)
        factor_scores = (
            rank(momentum, ascending=False) +
            rank(volatility, ascending=True) +  # We want low volatility
            rank(size, ascending=False)
        )

        return factor_scores