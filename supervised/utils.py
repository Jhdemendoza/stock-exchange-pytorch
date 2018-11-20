import collections
import datetime
import numpy as np
import pandas as pd
import os
import six
import scipy.stats as stats


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

    df.minute = df.minute.map(lambda x: datetime.datetime.strptime(x, '%H:%M'))
    df['hour'] = df.minute.map(lambda x: str(x.hour))

    return df


def get_numeric_categoric(df):
    numeric_cols, categorical_cols = [], []

    for col in df:
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


# Should be merged with the above function... for now...
def delta_dataframe_with_y_columns(df, numeric_columns):
    '''
    log numerical columns, then return deltas
    '''

    max_shift_forward = 20
    added_columns = []
    for shift in [-max_shift_forward, -10, -5, 3, 5, 10, max_shift_forward]:
        for col in numeric_columns:
            new_col_name = col + '_' + str(shift)
            df[new_col_name] = df[col].shift(shift)
            added_columns += [new_col_name]

    df[numeric_columns + added_columns] = df[numeric_columns + added_columns].apply(np.log)

    # for lookbacks
    for new_col in added_columns:
        original_col, added_part = new_col.split('_')
        df[new_col] = df[new_col] - df[original_col] if '-' in added_part else \
            df[original_col] - df[new_col]

    # for today
    # This line is necessary
    temp = df[numeric_columns] - df[numeric_columns].shift(1)
    df[numeric_columns] = temp

    assert (df.index == np.arange(len(df))).all()
    df.drop(df.index[list(range(max_shift_forward))], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    #                            negative max_shift_back...
    df.drop(index=list(range(len(df) - max_shift_forward, len(df))), inplace=True)

    return df


def train_df_test_df(ticker):
    def concat_and_return_csvs(original_df, ticker_files):
        for item in ticker_files[1:]:
            this_df = pd.read_csv(data_path + item)
            original_df = pd.concat([original_df, this_df])
        return original_df

    data_path = 'data/daily_data/'
    ticker_files = [item for item in os.listdir(data_path) if ticker in item.split('_')]
    ticker_files.sort()

    if len(ticker_files) == 0:
        return None, None

    split_idx = int(len(ticker_files) * 0.8)
    train_ticker_files, test_ticker_files = ticker_files[:split_idx], ticker_files[split_idx:]

    train_df = pd.read_csv(data_path + train_ticker_files[0])
    train_df = concat_and_return_csvs(train_df, train_ticker_files)

    test_df = pd.read_csv(data_path + test_ticker_files[0])
    test_df = concat_and_return_csvs(test_df, test_ticker_files)

    return train_df, test_df


def load_dataframes(ticker):
    train_df, test_df = train_df_test_df(ticker)

    if train_df is None:
        return None, None, None, None

    train_df = get_processed_minute_data(train_df)
    test_df = get_processed_minute_data(test_df)

    numeric_cols, categoric_cols = get_numeric_categoric(train_df)
    # This is for the time being...
    categoric_cols = ['weekday', 'month', 'hour']

    train_df = delta_dataframe_with_y_columns(train_df, numeric_cols)
    test_df = delta_dataframe_with_y_columns(test_df, numeric_cols)

    # Re-evaluate column names from the deltas
    numeric_cols, _ = get_numeric_categoric(train_df)

    return train_df, test_df, numeric_cols, categoric_cols


def get_y_cols(numeric_cols):
    price_cols = [item for item in numeric_cols if '-' in item]
    interested_cols = [item for item in price_cols if 'High' in item or 'Low' in item]
    not_interested_cols = list(set(price_cols) - set(interested_cols))
    return interested_cols, not_interested_cols
