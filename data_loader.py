import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        """Fetches daily pricing data from Yahoo Finance."""
        print(f"Fetching data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        return self.data

    def get_clean_data(self):
        """Cleans the data by forward-filling and dropping NaNs."""
        if self.data is None:
            self.fetch_data()
        
        # Notice the .loc before the brackets!
        data_clean = self.data.ffill().dropna().loc[self.start_date:self.end_date]
        
        # Bulletproof manual column extraction
        close_cols = []
        vol_cols = []
        tickers = []
        
        # Iterate over the raw column names (which are tuples like ('Close', 'AAPL'))
        for col in data_clean.columns:
            if isinstance(col, tuple):
                if 'Close' in col:
                    close_cols.append(col)
                    # The ticker is the other element in the tuple
                    tickers.append([c for c in col if c != 'Close'][0])
                elif 'Volume' in col:
                    vol_cols.append(col)
                    
        # Filter the dataframe to only those specific columns
        close = data_clean[close_cols]
        volume = data_clean[vol_cols]
        
        # Rename the columns to just the tickers (e.g., 'AAPL', 'MSFT')
        close.columns = tickers
        volume.columns = tickers
        
        return close, volume