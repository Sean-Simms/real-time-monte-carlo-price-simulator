import numpy as np
import yfinance as yf


def get_market_data(ticker, period="2y"):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No market data found for {ticker}")

    prices = data["Close"].dropna().squeeze()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    expected_return = float(log_returns.mean() * 252)
    volatility = float(log_returns.std() * np.sqrt(252))
    current_price = float(prices.iloc[-1])

    return data, current_price, expected_return, volatility