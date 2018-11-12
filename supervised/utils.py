import collections
import six
import scipy.stats as stats
import numpy as np
import pandas as pd
import datetime


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


def get_processed_minute_data(df):
    cols = df.columns.tolist()
    cols_to_drop = cols[:4] + ['label', 'changeOverTime', 'close', 'high',
                               'low', 'marketAverage', 'marketClose',
                               'marketOpen', 'volume', 'numberOfTrades',
                               'notional', 'open', 'marketChangeOverTime']
    df.drop(cols_to_drop, axis=1, inplace=True)
    # necessary
    df.reset_index(drop=True, inplace=True)

    idx_to_drop = df.index[df.marketNotional == 0.0]
    df.drop(idx_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.date = df.date.map(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))
    df['weekday'] = df.date.map(lambda x: str(x.weekday()))
    df['month'] = df.date.map(lambda x: str(x.month))

    df.minute = first_df.minute.map(lambda x: datetime.datetime.strptime(x, '%H:%M'))
    df['hour'] = first_df.minute.map(lambda x: str(x.hour))

    return df


def get_numeric_categoric(df):
    numeric_cols, categorical_cols = [], []

    for col in first_df:
        if np.issubdtype(df[col].dtype, np.number):
            numeric_cols += [col]
        else:
            categorical_cols += [col]

    return numeric_cols, categorical_cols


def delta_dataframe(df, numeric_columns):
    '''
    :param df:
    :param numeric_columns:
    :return: np.log(deltas)
    '''
    added_columns = []
    for shift in [3, 5, 10, 20]:
        for col in numeric_columns:
            new_col_name = col + '_' + str(shift)
            df[new_col_name] = df[col].shift(shift)
            added_columns += [new_col_name]

    df[numeric_columns + added_columns] = df[numeric_columns + added_columns].apply(np.log)

    # for lookbacks
    for new_col in added_columns:
        original_col = new_col.split('_')[0]
        df[new_col] = df[original_col] - df[new_col]

    # for today
    # This line is necessary
    temp = df[numeric_columns] - df[numeric_columns].shift(1)
    df[numeric_columns] = temp

    assert (df.index == np.arange(len(first_df))).all()
    df.drop(df.index[list(range(20))], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
