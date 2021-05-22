'''
Plot average returns vs volatility (efficient market frontier)
of many possible portfolio combinations

'''

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


# download data
def get_stock_data(tickers: list) -> pd.DataFrame:
    stocks = pd.DataFrame()
    for ticker in tickers:
        data = yf.Ticker(ticker).history(period='5y')
        stocks[ticker] = np.log(data['Close'] / data['Close'].shift(1))
    return stocks


# get weights
def get_weights(size: Union[int, tuple]) -> np.ndarray:
    weights = np.random.uniform(0, 1, size=size)
    weights /= weights.sum()
    return weights


# yearly portfolio performance
# https://github.com/GuidoBR/python-for-finance/tree/master/python-for-finance-investment-fundamentals-data-analytics/
def get_performance(stocks: pd.DataFrame, weights: np.ndarray) -> tuple:
    returns = np.sum(weights * stocks.mean()) * 250
    volatility = np.sqrt(np.dot(weights.T, np.dot(stocks.cov() * 250, weights)))
    sharpe = returns / volatility
    return returns, volatility, sharpe


# efficient market frontier
def main() -> None:

    # get data
    tickers = [
        'GOOG',
        'AMZN',
        'AAPL',
        'NFLX',
        'MSFT',
        'TSLA',
        'AMC',
        'GME',
        'NVDA',
        'TECH',
        'INTC',
        'BABA',
        'PYPL',
        'CSCO',
        'MTCH',
        'ADBE',
        'DBX',
        'QQQ',
        'CRM',

    ]
    stocks = get_stock_data(tickers)

    # make portfolios
    port_perf = []
    for _ in range(1000):
        weights = get_weights(len(tickers))
        perf = get_performance(stocks, weights)
        port_perf.append(list(perf))

    cols = ['returns', 'volatility', 'sharpe']
    port_perf = pd.DataFrame(port_perf, columns=cols)

    # plot emf
    plt.scatter(
        port_perf['volatility'],
        port_perf['returns'],
        c=port_perf['sharpe'],
        cmap='viridis'
    )
    plt.colorbar(label='Sharpe')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()