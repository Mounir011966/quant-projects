import yfinance as yf
import pandas as pd

def download_data(tickers, start_date="2020-01-01", end_date="2024-01-01"):
    data = yf.download(tickers, start=start_date, end=end_date)[["Close"]]
    data.columns = tickers
    data = data.dropna()
    return data

if __name__ == "__main__":
    tickers = ["XOM", "CVX"]  # ExxonMobil et Chevron
    df = download_data(tickers)
    df.to_csv("data/price_data.csv")
    print(" Données sauvegardées dans data/price_data.csv")
