import yfinance as yf
import numpy as np

def get_market_data(ticker, period="2y"):
    data = yf.download(ticker,period=period,auto_adjust=True,progress=False)
    if data.empty:
        raise ValueError(f"No market data found for {ticker}")
    prices = data["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    expected_return = log_returns.mean().item() * 252
    volatility = log_returns.std().item() * np.sqrt(252)
    current_price = prices.iloc[-1].item()
    return data, current_price, expected_return, volatility
