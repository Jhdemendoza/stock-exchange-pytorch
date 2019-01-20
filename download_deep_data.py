import argparse
import datetime
import glob
import math
import pandas as pd
import pandas_datareader.data as web
import re
import time
from functools import partial

from download_daily_data import all_tickers, BASE_URL, my_list, get_dataframe


# Looks bad.. but for now
my_list = list(my_list)
snp = pd.read_csv('data/snp_tickers.csv').Symbol.tolist()


def get_tickers_in_tuple():
    ticker_tuples = []
    for idx in range(math.ceil(len(my_list) / 10)):
        this_tuple = my_list[idx*10:(idx+1)*10]
        this_tuple = ','.join(this_tuple)
        ticker_tuples += [this_tuple]
    return ticker_tuples


def _collect_deep(so_far):
    # Might make this async...
    # But this is tolerable, since one second delay makes some room for a rest time
    # For Async reference:
    #     https://medium.freecodecamp.org/a-guide-to-asynchronous-programming-in-python-with-asyncio-232e2afa44f6
    #     https://stackoverflow.com/questions/51471848/async-request-for-python
    def _get_deep_book():
        _this_row = []
        for tickers in get_tickers_in_tuple():
            my_url = BASE_URL + '/deep/book?symbols={}'.format(tickers)
            results = get_dataframe(my_url)
            if results is not None:
                _this_row += [results]
        return pd.concat(_this_row, axis=1)

    this_row = _get_deep_book()

    if so_far is None:
        so_far = this_row
    else:
        so_far = pd.concat([so_far, this_row], axis=0)

    return so_far


def _collect_top(so_far, ticker_list):
    time.sleep(1)
    this_row = web.DataReader(ticker_list, 'iex-tops')

    if so_far is None:
        so_far = this_row
    else:
        so_far = pd.concat([so_far, this_row], axis=0)

    return so_far


def collect_wrapper(collect_fn):
    # Initializing to pd.DataFrame() might be a more sound idea...
    so_far = None

    try:
        while True:
            now = datetime.datetime.now().__str__()
            pattern = re.compile(r'(\d{2}):(\d{2})')
            current_hour = pattern.search(now).group()

            trading_status_url = '/deep/trading-status?symbols=snap'
            trading_status = get_dataframe(BASE_URL + trading_status_url)

            if trading_status is not None and not trading_status.empty:
                trading_status = trading_status.T['status'][0] == 'T'

            print('*** Collecting data, current time: {}'
                  .format(current_hour))

            if current_hour < '09:30':
                print('***     Going to sleep for 1 minute... ')
                time.sleep(60)

            elif (current_hour < '16:00') and trading_status:
                so_far = collect_fn(so_far)

            else:
                print('Breaking the loop: {}, {}'
                      .format(current_hour, trading_status))
                break
    except KeyboardInterrupt:
        print('KeyBoard Interrupt Raised!')
    except Exception as e:
        print(e.message, e.args)
    finally:
        return so_far


def get_args():
    parser = argparse.ArgumentParser(description='Choose deep vs top')
    parser.add_argument('--collect', choices=['deep', 'top'], required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    collect = args.collect

    today = datetime.datetime.today().date().__str__()
    print('--- {} book being collected... {}'.format(collect, today))

    starting_time = time.time()

    if args.collect == 'deep':
        data = collect_wrapper(_collect_deep)
    else:
        _collect_top = partial(_collect_top, ticker_list=snp)
        data = collect_wrapper(_collect_top)

    if data is not None:
        print('--- Saving... {}'.format(datetime.datetime.now()))
        today = today.replace('-', '_')
        # Might want to add mkdir...
        file_name = 'data/{}/{}_data_{}'.format(collect, collect, today)

        # while file exists, add _
        while glob.glob(file_name):
            file_name += '_'

        data.to_csv(file_name)

    print('--- Finished collecting, took: {:.3f} seconds'
          .format(time.time()-starting_time))

