import datetime
import glob
import math
import pandas as pd
import time


from download_daily_data import all_tickers, BASE_URL, my_list, get_dataframe


# Looks bad.. but for now
my_list = list(my_list)
def get_tickers_in_tuple():
    ticker_tuples = []
    for idx in range(math.ceil(len(my_list) / 10)):
        this_tuple = my_list[idx * 10:(idx + 1) * 10]
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
            _this_row += [results]
        return pd.concat(_this_row, axis=1)

    this_row = _get_deep_book()

    if so_far is None:
        so_far = this_row
    else:
        so_far = pd.concat([so_far, this_row], axis=0)

    return so_far


def collect_deep():
    so_far = None

    try:
        while True:
            now = datetime.datetime.now()
            # Might as well regex?
            current_hour = now.__str__().split(' ')[1].split('.')[0][:5]

            trading_status_url = '/deep/trading-status?symbols=snap'
            trading_status = get_dataframe(BASE_URL + trading_status_url)

            if trading_status is not None and not trading_status.empty:
                trading_status = trading_status.T['status'][0] == 'T'

            if '09:29' > current_hour:
                print('*** Collecting data, current time: {}'.format(current_hour))
                print('***     Going to sleep for 1 minute... ')
                time.sleep(60)

            elif (current_hour < '16:00') and trading_status:
                so_far = _collect_deep(so_far)

            else:
                print('Breaking the loop: {}, {}'.format(current_hour, trading_status))
                break
    except KeyboardInterrupt:
        print('KeyBoard Interrupt Raised!')
    except Exception as e:
        print(e.message, e.args)
    finally:
        return so_far


if __name__ == '__main__':
    today = datetime.datetime.today().date().__str__()
    print('--- Deep Book being collected... {}'.format(today))

    starting_time = time.time()
    data = collect_deep()

    if data is not None:
        print('--- Saving... {}'.format(datetime.datetime.now()))
        today = today.replace('-', '_')
        file_name = 'data/deep/deep_data_{}'.format(today)
        # while file exists, add _
        while glob.glob(file_name):
            file_name += ['_']

        data.to_csv(file_name)

    print('--- Finished collecting, took: {:.3f} seconds'.format(
        time.time()-starting_time))



