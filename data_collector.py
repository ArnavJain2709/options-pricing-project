import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# --- Configuration ---
# 1. List of stock tickers with active options markets
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN']

# 2. Set the risk-free interest rate. 
# For a student project, using a constant is fine. 
# A good proxy is the 3-month US Treasury Bill yield (~5% in late 2025).
RISK_FREE_RATE = 0.05

# 3. Output file to store the data
OUTPUT_CSV = 'options_dataset.csv'
# --------------------


def fetch_options_data(ticker_symbol):
    """
    Fetches the entire options chain for a given stock ticker.
    """
    stock = yf.Ticker(ticker_symbol)
    
    # Get the current stock price
    try:
        current_stock_price = stock.history(period='1d')['Close'].iloc[0]
    except IndexError:
        print(f"Could not fetch current price for {ticker_symbol}. Skipping.")
        return pd.DataFrame()

    all_options_data = []

    # Iterate through all available expiration dates
    for exp_date in stock.options:
        options_chain = stock.option_chain(exp_date)
        
        calls = options_chain.calls
        calls['optionType'] = 'call'
        puts = options_chain.puts
        puts['optionType'] = 'put'

        # Combine calls and puts
        options_df = pd.concat([calls, puts])
        
        # Add extra info
        options_df['expirationDate'] = exp_date
        options_df['ticker'] = ticker_symbol
        options_df['timestamp'] = datetime.now()
        options_df['underlyingPrice'] = current_stock_price
        
        all_options_data.append(options_df)

    if not all_options_data:
        return pd.DataFrame()
        
    return pd.concat(all_options_data, ignore_index=True)


def process_and_clean_data(df):
    """
    Processes the raw options data to create features for the ML model.
    """
    # Calculate the midpoint of the bid-ask spread for the option price
    df['marketPrice'] = (df['bid'] + df['ask']) / 2.0

    # Calculate Time to Expiration in years
    df['expirationDate'] = pd.to_datetime(df['expirationDate'])
    df['daysToExpiration'] = (df['expirationDate'] - df['timestamp']).dt.days
    df['timeToExpiration'] = df['daysToExpiration'] / 365.0
    
    # Calculate Moneyness
    df['moneyness'] = df['underlyingPrice'] / df['strike']

    # Add risk-free rate
    df['riskFreeRate'] = RISK_FREE_RATE
    
    # Filter out contracts with no bid/ask, as they are illiquid
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]

    # Select the final columns needed for the model
    # We now keep the extra columns needed for the Black-Scholes calculation
    final_cols = [
        'underlyingPrice', # Keep this for S
        'strike',          # Keep this for K
        'optionType',      # Keep this for call/put
        'moneyness', 
        'timeToExpiration', 
        'impliedVolatility', 
        'riskFreeRate', 
        'marketPrice'       # Target variable y
    ]
    
    processed_df = df[final_cols].copy()
    
    # Drop rows with any missing values
    processed_df.dropna(inplace=True)
    
    return processed_df


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting options data collection...")
    
    all_ticker_data = []
    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        data = fetch_options_data(ticker)
        if not data.empty:
            all_ticker_data.append(data)

    if not all_ticker_data:
        print("No data collected. Exiting.")
    else:
        # Combine data from all tickers into one DataFrame
        raw_data_df = pd.concat(all_ticker_data, ignore_index=True)
        
        # Process the raw data to create model-ready features
        model_ready_df = process_and_clean_data(raw_data_df)
        
        # Append data to the CSV file
        if os.path.exists(OUTPUT_CSV):
            model_ready_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            print(f"Appended {len(model_ready_df)} new records to {OUTPUT_CSV}")
        else:
            model_ready_df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False)
            print(f"Created {OUTPUT_CSV} and saved {len(model_ready_df)} records.")