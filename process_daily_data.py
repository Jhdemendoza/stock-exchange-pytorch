import requests
import pandas as pd
import numpy as np
import time
import os.path


BASE_URL = 'https://api.iextrading.com/1.0/'
STOCK = 'stock/'
NEWS = '/news/last/10'


def get_dataframe(address):
    response = requests.get(address)
    try:
        df = pd.DataFrame.from_dict(response.json(), orient='columns')
    except Exception as e:
        print('Exception raised: ', address, e)
        df = None
    return df


if __name__ == '__main__':

    ref_symbols = BASE_URL + 'ref-data/symbols'
    symbols_df = get_dataframe(ref_symbols)

    for ticker in symbols_df.symbol:

        ticker = str.lower(ticker)
        # price data
        address = BASE_URL + STOCK + '{}/chart/1d'.format(ticker)
        ticker_df = get_dataframe(address)

        if ticker_df is not None and 'minute' in ticker_df.columns and 'date' in ticker_df.columns:
            dates = ticker_df.date.unique()
            # It's better to check the length, nan usually means complete junk
            if len(dates) > 1:
                print('{}: unique dates greater than two: {}'.format(ticker, dates))
                continue
            today = dates.item()
            ticker_df.to_csv('data/daily_data/{}'.format(today+'_'+ticker))

            news_address = BASE_URL + STOCK + ticker + NEWS
            news_df = get_dataframe(news_address)
            news_df.to_csv('data/news_data/{}'.format(today+'_'+ticker+'_news'))

        time.sleep(0.1)
