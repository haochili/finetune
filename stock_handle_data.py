import yfinance as yf
import pandas_ta as ta
import numpy as np

# Define the ticker symbol and the period
ticker_symbol = "SPY"
start_date = '1973-01-01'  # Adjust based on how far back data is available
#start_date = '2024-01-01'  # Adjust based on how far back data is available
end_date = '2024-05-08'    # Use the current date or the end of your analysis period
macd_key = 26
# Attempt to fetch the historical data from Yahoo Finance
try:
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
    if data.empty:
        raise ValueError("No data fetched. Please check the ticker symbol or date range.")
except Exception as e:
    print(f"Failed to download data: {e}")
    exit()

# Save the downloaded data to a CSV file
data.to_csv('data/SPY_daily_data.csv')


# Continue with processing only if data is present
if not data.empty:
    # data['unixtime'] = data.index.astype(np.int64) // 10**9
    # data['unixtimesq'] = data['unixtime'] ** 2
    # data['time_from_now'] = data['unixtime'].iloc[-1] - data['unixtime']
    # data['time_from_now_sq'] = data['time_from_now'] ** 2

    for i in range(1, macd_key):
        data[f'prev_{i}_day'] = data['Adj Close'].shift(i)

    # Calculating the change in price
    data['delta'] = data['Adj Close'].pct_change()

    # Adding technical indicators using pandas_ta
    data['rsi'] = ta.rsi(data['Adj Close'])
    data['ema'] = ta.ema(data['Adj Close'], length=14)
    data['cmf'] = ta.cmf(data['High'], data['Low'], data['Adj Close'], data['Volume'])
    #data['vwap'] = ta.vwap(data['High'], data['Low'], data['Adj Close'], data['Volume'])
    data['bollinger_high'] = ta.bbands(data['Adj Close'])['BBL_5_2.0']
    data['bollinger_low'] = ta.bbands(data['Adj Close'])['BBU_5_2.0']
    data['macd'] = ta.macd(data['Adj Close'])[f'MACD_12_{macd_key}_9']
    data['tomr'] = data['Adj Close'].shift(-1)
 
    # # Stochastic oscillator
    # data['stoch_k'], data['stoch_d'] = ta.stoch(data['High'], data['Low'], data['Adj Close'])

    # Preparing the data by removing NA values
    data.dropna(inplace=True)

    # Display the final dataframe
    print(data.head())
else:
    print("Dataframe is empty. No data to process.")

data.to_csv('data/SPY_daily_data_with_indicators.csv')


