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


def get_numeric_categoric(df):
    numeric_cols, categoric_cols = [], []

    for col in df:
        if np.issubdtype(df[col].dtype, np.number):
            numeric_cols += [col]
        else:
            categoric_cols += [col]

    return numeric_cols, categoric_cols


# --------------------------------------------------------------------
# --- minute-data
# --------------------------------------------------------------------
def get_processed_minute_data(dataframes_in_list):
    for df in dataframes_in_list:
        cols = df.columns.tolist()
        cols_to_drop = cols[:4] + ['label',
                                   'open',
                                   'close',
                                   'low',
                                   'high',
                                   'volume',
                                   'notional',
                                   'numberOfTrades',
                                   'changeOverTime',
                                   'marketChangeOverTime',
                                   'marketAverage',
                                   'marketClose',
                                   'marketNotional']

        df.drop(cols_to_drop, axis=1, inplace=True)
        # necessary
        df.reset_index(drop=True, inplace=True)

        df.date = df.date.map(lambda x: datetime.datetime
                                                .strptime(str(x), '%Y%m%d'))
        df['weekday'] = df.date.map(lambda x: str(x.weekday()))
        df['month'] = df.date.map(lambda x: str(x.month))
        df['hour'] = df.minute.map(lambda x: str(datetime.datetime
                                                         .strptime(x, '%H:%M')
                                                         .hour))

        df.replace(-1.0, np.NaN, inplace=True)
        df['marketNumberOfTrades'].replace(0, np.NaN, inplace=True)
        df['marketVolume'].replace(0, np.NaN, inplace=True)
        df.interpolate(limit_direction='Both', inplace=True)

    return dataframes_in_list


# --------------------------------------------------------------------
# --- minute-data
# --------------------------------------------------------------------
def delta_dataframe_from_a_list(df_lists, numeric_columns, args=None):
    return [delta_dataframe_with_y_columns_new(df, numeric_columns, args)
            for df in df_lists]


# --------------------------------------------------------------------
# --- minute-data
# --------------------------------------------------------------------
def delta_dataframe_with_y_columns_new(df, numeric_columns, args=None):
    '''
    log numerical columns, then return deltas
    '''

    added_columns = []

    if args is not None:
        min_shift_forward = args.min_shift_forward
        max_shift_forward = args.max_shift_forward
        increment = args.shift_increment
        target_shift = args.target_shift
    else:
        min_shift_forward, max_shift_forward, increment, target_shift = 4, 30, 6, 10

    # Just do this because it makes our life easier...
    shift_dates = list(range(-max_shift_forward, -min_shift_forward, increment))
    shift_dates = list(map(lambda x: -x, reversed(shift_dates))) + [-target_shift]
    for shift in shift_dates:
        for col in numeric_columns:
            new_col_name = col + '_' + str(shift)
            df[new_col_name] = df[col].shift(shift)
            added_columns += [new_col_name]

    df[numeric_columns + added_columns] = df[numeric_columns + added_columns].apply(np.log)

    # for lookbacks
    for new_col in added_columns:
        original_col, added_part = new_col.split('_')
        df[new_col] = (df[new_col] - df[original_col] if '-' in added_part else
                       df[original_col] - df[new_col])

    # for today
    # This line is necessary
    temp = df[numeric_columns] - df[numeric_columns].shift(1)
    df[numeric_columns] = temp

    assert (df.index == np.arange(len(df))).all(), '{} vs np.arange...'.format(df.index)

    #                            negative max_shift_back...
    df.drop(index=list(range(len(df) - target_shift, len(df))), inplace=True)
    df.drop(index=list(range(max_shift_forward)), inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# --------------------------------------------------------------------
# --- minute-data
# --------------------------------------------------------------------
def train_df_test_df_in_lists(ticker, date_starting, date_ending):

    def _get_dataframes_in_a_list(ticker_files):
        return [pd.read_csv(data_path+ticker) for ticker in ticker_files]

    # daily_data := daily minute data... misnomer
    data_path = 'data/daily_data/'
    ticker_files = [file for file in os.listdir(data_path)
                    if (ticker in file.split('_') and (date_starting < file < date_ending))]
    ticker_files.sort()

    if len(ticker_files) == 0:
        return None, None

    split_idx = int(len(ticker_files) * 0.8)
    train_ticker_files, test_ticker_files = ticker_files[:split_idx], ticker_files[split_idx:]

    train_dataframes_in_list = _get_dataframes_in_a_list(train_ticker_files)
    test_dataframes_in_list = _get_dataframes_in_a_list(test_ticker_files)

    return train_dataframes_in_list, test_dataframes_in_list


# --------------------------------------------------------------------
# --- minute-data
# --------------------------------------------------------------------
def load_dataframes(ticker, args=None):
    train_df_a_list, test_df_a_list = train_df_test_df_in_lists(ticker,
                                                                args.date_starting,
                                                                args.date_ending)

    if train_df_a_list is None:
        return None, None, None, None

    train_df_a_list = get_processed_minute_data(train_df_a_list)
    test_df_a_list = get_processed_minute_data(test_df_a_list)

    sample_df = train_df_a_list[0]

    numeric_cols, categoric_cols = get_numeric_categoric(sample_df)
    # This is for the time being...
    categoric_cols = ['weekday', 'month', 'hour']

    train_df_a_list = delta_dataframe_from_a_list(train_df_a_list, numeric_cols, args)
    test_df_a_list = delta_dataframe_from_a_list(test_df_a_list, numeric_cols, args)

    # Re-evaluate column names from the deltas
    numeric_cols, _ = get_numeric_categoric(sample_df)

    train_df = pd.concat(train_df_a_list).reset_index(drop=True)
    test_df = pd.concat(test_df_a_list).reset_index(drop=True)

    return train_df, test_df, numeric_cols, categoric_cols


# --------------------------------------------------------------------
# --- ohlc for one data point per day
# --------------------------------------------------------------------
# So much duplicate code... but wtf...
def ohlc_train_df_test_df(ticker, args=None):
    first_df = get_original_df(ticker)
    num_cols = first_df.columns.tolist()
    num_cols.remove('date')

    if args is not None and args.create_more_features:
        first_df = get_features(first_df, args, num_cols)
        num_cols, _ = get_numeric_categoric(first_df)

    split_idx = int(len(first_df) * 0.8)

    train_df = first_df.iloc[:split_idx].copy()
    test_df = first_df.iloc[split_idx:].copy()
    test_df.reset_index(drop=True, inplace=True)

    train_df = delta_dataframe_with_y_columns_new(train_df, num_cols, args)
    test_df = delta_dataframe_with_y_columns_new(test_df, num_cols, args)

    num_cols, cat_cols = get_numeric_categoric(train_df)

    return train_df, test_df, num_cols, cat_cols


def get_y_cols(numeric_cols, minute_data=False):
    price_cols = [item for item in numeric_cols if '-' in item]
    high = 'High' if minute_data else 'high'
    low = 'Low' if minute_data else 'low'

    interested_cols = [item for item in price_cols if high in item or low in item]
    not_interested_cols = list(set(price_cols) - set(interested_cols))
    return interested_cols, not_interested_cols


# --------------------------------------------------------------------
# --- ohlc for one data point per day
# --------------------------------------------------------------------
def get_original_df(ticker):
    first_df = pd.read_csv('data/ohlc/{}'.format(ticker))
    if len(first_df) < (252 * 5 - 2):
        print('Length insufficient: length of {}'.format(len(first_df)))
        return None, None, None, None
    # first_df.set_index('date', drop=True, inplace=True)
    first_df.drop(first_df.columns[0], axis=1, inplace=True)
    first_df.drop(['label',
                   'unadjustedVolume',
                   'changeOverTime',
                   'change',
                   'vwap',
                   'changePercent'], axis=1, inplace=True)
    return first_df


# --------------------------------------------------------------------
# --- ohlc for one data point per day
# --------------------------------------------------------------------
def get_features(df, args, numeric_cols):
    window_max = args.max_shift_forward
    window_min = (args.target_shift if args.target_shift < args.min_shift_forward
                  else args.min_shift_forward)
    window_mid = (window_min + window_max) // 2

    for col in numeric_cols:
        # Do not use underscore `_` because delta_dataframe_fn looks for it!!!
        for item in [window_min, window_mid, window_max]:
            name = col + str(item) + 'Ma'
            df[name] = df[col].rolling(window=item).mean()
            df[name + 'Std'] = df[col].rolling(window=item).std()

    # Although we drop max_shift_forward in the delta function,
    #     delta function still requires dropping early on to not make a mess
    df.drop(index=list(range(window_max)), inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
