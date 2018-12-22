import requests
import pandas as pd
import numpy as np
import time
import os.path


BASE_URL = 'https://api.iextrading.com/1.0'
STOCK = '/stock'
NEWS = '/news/last/10'
my_list = {
     'tsla', 'aapl', 'goog', 'sbux', 'tm', 'msft', 'gs', 'fb',
     'bcs', 'atvi', 'baba', 'virt', 'nvda', 'amd',
     'tot', 'bp', 'akam', 't', 'vz', 'ntdoy', 'shak',
     'cmg', 'ddd', 'ssys', 'cost', 'cg', 'amzn', 'bidu',
     'ntes', 'sina', 'snap', 'nvs', 'adbe', 'orcl', 'cldr',
     'habt', 'aep', 'duk', 'd', 'peg', 'pcg',
     'pg', 'wm', 'tsn', 'gm', 'mu', 'pfe', 'brk.b', 'bf.a', 'lgf.a',
     'spy',
}

etf_tickers = set(pd.read_csv('data/etf_list')['NAME'][:30])

russell_tickers = pd.read_csv('data/russell1000.csv').Ticker.tolist()
russell_ticker_set = set(map(lambda x: str.lower(x), russell_tickers))

all_tickers = my_list | etf_tickers | russell_ticker_set


def get_dataframe(address):
    response = requests.get(address)
    try:
        df = pd.DataFrame.from_dict(response.json(), orient='columns')
    except Exception as e:
        print('Exception raised: ', address, e)
        df = None
    return df


def save_df(ticker_df, ticker):
    dates = ticker_df.date.unique()
    # It's better to check the length, nan usually means complete junk
    if len(dates) > 1:
        print('{}: unique dates greater than two: {}'.format(ticker, dates))
        return
    today = dates.item()
    ticker_df.to_csv('data/daily_data/{}'.format(today+'_'+ticker))

    news_address = BASE_URL + STOCK + '/' + ticker + NEWS
    news_df = get_dataframe(news_address)
    news_df.to_csv('data/news_data/{}'.format(today+'_'+ticker+'_news'))


def filling_in_for_missing_data():
    '''
    Use only when missing some days in the past month
    '''
    one_month = BASE_URL + STOCK + '/aapl/chart/1m'
    d1 = get_dataframe(one_month)
    dates = d1.date.tolist()
    dates = [''.join(x.split('-')) for x in dates]

    # all_tickers = my_list | russell_ticker_set

    for date in dates:
        for ticker in all_tickers:
            address = 'https://api.iextrading.com/1.0/stock/{}/chart/date/{}'.format(
                ticker, date)
            ticker_df = get_dataframe(address)
            if ticker_df is not None and 'minute' in ticker_df.columns and 'date' in ticker_df.columns:
                dates = ticker_df.date.unique()
                # It's better to check the length, nan usually means complete junk
                if len(dates) > 1:
                    print('{}: unique dates greater than two: {}'.format(ticker, dates))
                    continue
                today = dates.item()
                ticker_df.to_csv('data/daily_data/{}'.format(today+'_'+ticker))


def download_one_year_ohlc():
    for ticker in all_tickers:
        address = 'https://api.iextrading.com/1.0/stock/{}/chart/5y'.format(ticker)
        ticker_df = get_dataframe(address)
        if ticker_df is not None and 'close' in ticker_df.columns and 'high' in ticker_df.columns:
            dates = ticker_df.date.unique()
            # It's better to check the length, nan usually means complete junk
            if len(dates) < 1:
                print('{}: unique dates less than one: {}'.format(ticker, dates))
                continue
            ticker_df.to_csv('data/ohlc/{}'.format(ticker))


if __name__ == '__main__':

    for ticker in all_tickers:

        # price data
        address = BASE_URL + STOCK + '/{}/chart/1d'.format(ticker)
        ticker_df = get_dataframe(address)

        if ticker_df is not None and 'minute' in ticker_df.columns and 'date' in ticker_df.columns:
            save_df(ticker_df, ticker)

        time.sleep(0.1)
