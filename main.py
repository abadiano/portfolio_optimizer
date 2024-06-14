import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Updated list of stock tickers
STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JNJ', 'V', 'PG',
    'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CMCSA', 'VZ', 'INTC',
    'PFE', 'MRK', 'ABT', 'ABBV', 'LLY', 'TMO', 'DHR', 'BMY', 'MDT', 'CVS',
    'PEP', 'KO', 'MCD', 'NKE', 'SBUX', 'COST', 'WMT', 'TGT', 'LOW', 'HD',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BK', 'BLK', 'SCHW',
    'XOM', 'CVX', 'COP', 'PSX', 'VLO', 'MPC', 'OXY', 'SLB', 'HAL', 'BKR',
    'BA', 'RTX', 'LMT', 'NOC', 'GD', 'HII', 'TXT', 'LHX', 'TDG', 'HEI',
    'CSCO', 'IBM', 'ORCL', 'ACN', 'SAP', 'ADP', 'INTU', 'CRM', 'NOW', 'WDAY',
    'ADSK', 'SNPS', 'ANSS', 'MSCI', 'CDNS', 'AKAM', 'FTNT', 'NET', 'CRWD', 'ZS'
]

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period='1y')
    if hist.empty:
        return None
    return {
        'symbol': ticker,
        'companyName': info.get('longName', ''),
        'marketCap': info.get('marketCap', 0),
        'sector': info.get('sector', ''),
        'previousClose': info.get('previousClose', 0),
        'recommendation': info.get('recommendationKey', ''),
        'history': hist
    }

def fetch_all_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        data = fetch_stock_data(ticker)
        if data:
            stock_data.append(data)
        else:
            print(f"{ticker}: No data found, symbol may be delisted or invalid.")
    return stock_data

def calculate_historical_performance(hist):
    if hist.empty:
        return np.nan
    return (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]

def calculate_volatility(hist):
    if hist.empty:
        return np.nan
    return np.std(hist['Close'].pct_change()) * np.sqrt(252)

def add_performance_metrics(stocks):
    for stock in stocks:
        stock['historicalReturn'] = calculate_historical_performance(stock['history'])
        stock['volatility'] = calculate_volatility(stock['history'])
    return stocks


def filter_and_rank_stocks(stocks):
    # Example criteria for filtering: Sector diversification and volatility
    filtered_stocks = [stock for stock in stocks if
                       stock['sector'] in ['Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary',
                                           'Communication Services']]

    # Rank stocks based on criteria (e.g., historical return, volatility)
    ranked_stocks = sorted(filtered_stocks, key=lambda x: (x['historicalReturn'], -x['volatility']), reverse=True)
    return ranked_stocks[:10]  # Select top 10 stocks

def calculate_expected_returns_and_cov_matrix(stocks):
    prices = pd.DataFrame({stock['symbol']: stock['history']['Close'] for stock in stocks})
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean() * 252  # Annualize the returns
    cov_matrix = returns.cov() * 252  # Annualize the covariance matrix
    return expected_returns, cov_matrix


def portfolio_optimization(expected_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(expected_returns)

    def portfolio_performance(weights):
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, volatility

    def neg_sharpe_ratio(weights):
        returns, volatility = portfolio_performance(weights)
        return -(returns - risk_free_rate) / volatility

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets, ]

    result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def main():
    # Step 1: Fetch data for predefined stocks
    stocks = fetch_all_stock_data(STOCKS)

    # Step 2: Add performance metrics
    stocks = add_performance_metrics(stocks)

    # Step 3: Filter and rank stocks
    filtered_stocks = filter_and_rank_stocks(stocks)

    # Step 4: Calculate expected returns and covariance matrix
    expected_returns, cov_matrix = calculate_expected_returns_and_cov_matrix(filtered_stocks)

    # Step 5: Optimize portfolio
    optimal_weights = portfolio_optimization(expected_returns, cov_matrix)

    # Step 6: Display recommended portfolio and allocations
    print("Recommended Portfolio:")
    for stock, weight in zip(filtered_stocks, optimal_weights):
        print(f"Symbol: {stock['symbol']}, Name: {stock['companyName']}, Allocation: {weight:.2%}")


if __name__ == "__main__":
    main()
