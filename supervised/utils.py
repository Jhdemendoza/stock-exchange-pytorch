import collections
import six
import scipy.stats as stats
import numpy as np
import pandas as pd


def iterable(arg):
    return (isinstance(arg, collections.Iterable) and not
            isinstance(arg, six.string_types))


def print_distribution(output):
    if output.dim() != 2:
        return
    out = output.detach().cpu().numpy()
    out = out.reshape(-1, out.shape[-1])
    print('\r{} '.format(stats.describe(out)), end='')


def sin_lr(x):
    return np.abs(np.sin((x + 0.01) * 0.2))


def read_csv(ticker, is_etf=False):
    etf_path = 'iexfinance/Data/ETFs/{}.us.txt'
    stock_path = 'iexfinance/Data/Stocks/{}.us.txt'
    path = etf_path.format(ticker) if is_etf else stock_path.format(ticker)
    cur_df = pd.read_csv(path)
    cur_df.drop(['Volume', "OpenInt"], axis=1, inplace=True)
    return cur_df


def give_delta_historical(df):
    original_col = df.columns[1:]

    # shift days
    for shift_idx in [3, 5, 10, 20, 40]:
        for col in original_col:
            df[col + '_' + str(shift_idx)] = df[col].shift(shift_idx)

    # np.log all values
    for item in df:
        if np.issubdtype(df[item].dtype, np.number):
            df[item] = np.log(df[item])

    columns = df.columns
    df_values = df.values
    # for lookback
    for row_idx in range(5, df_values.shape[1], 4):
        df_values[:, row_idx:row_idx + 4] = df_values[:, 1:5] - \
                                            df_values[:, row_idx:row_idx + 4]
    # for today
    for idx in range(len(df_values) - 1, 0, -1):
        df_values[idx, 1:5] -= df_values[idx - 1, 1:5]

    return pd.DataFrame(df_values, columns=columns)


def process_output_data(spy_original):
    original_columns = spy_original.columns[1:]
    # shift 10, 5
    for col in original_columns:
        spy_original['10_'+str(col)] = spy_original[col].shift(-10)
    for col in original_columns:
        spy_original['5_'+str(col)] = spy_original[col].shift(-5)
    # log_delta
    for col in original_columns:
        spy_original['10_'+col] = np.log(spy_original['10_'+col].values) - \
                                  np.log(spy_original[col].values)
    for col in original_columns:
        spy_original['5_'+col] = np.log(spy_original['5_'+col].values) - \
                                 np.log(spy_original[col].values)
    return spy_original


def prepare_data(ticker, is_etf=False):
    # copy may actually be shallow, this is safe
    ticker_df_x, ticker_df_y = read_csv(ticker, is_etf), read_csv(ticker, is_etf)

    ticker_df_x = give_delta_historical(ticker_df_x)
    ticker_df_x.drop(list(range(40)), axis=0, inplace=True)
    ticker_df_x.iloc[:, 1:] = ticker_df_x.iloc[:, 1:].astype(np.float64, copy=False)

    ticker_df_y = process_output_data(ticker_df_y)
    y_column = ticker_df_y['10_Open'][40:]

    delete_from_back = y_column.isna().sum()
    ticker_df_x.drop(list(range(len(ticker_df_x)-delete_from_back, len(ticker_df_x))),
                     inplace=True)
    y_column = y_column[:-delete_from_back]

    return ticker_df_x, y_column
