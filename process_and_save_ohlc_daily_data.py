import pickle
from collections import defaultdict
from download_daily_data import my_list
from supervised import ohlc_get_y_cols, ohlc_train_df_test_df

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, QuantileTransformer
from sklearn.pipeline import FeatureUnion


def get_transfomed_combiner(df):
    # Use only the ones worked well in autoencoder
    transformer = [
        ('Data after min-max scaling',
         MinMaxScaler()),
        ('Data after max-abs scaling',
         MaxAbsScaler()),
        ('Data after quantile transformation (uniform pdf)',
         QuantileTransformer(output_distribution='uniform')),
        ('Data after sample-wise L2 normalizing',
         Normalizer()),
    ]

    combined = FeatureUnion(transformer)
    _ = combined.fit(df)

    return combined


def get_input_target(ticker):
    # messy code...
    train_df_original, test_df_original, numeric_cols, categoric_cols = ohlc_train_df_test_df(ticker)
    if train_df_original is None:
        return None, None, None, None

    y_cols, not_interested = ohlc_get_y_cols(numeric_cols)
    numeric_cols = list(sorted(set(numeric_cols) - set(y_cols) - set(not_interested)))

    train_df, y_train = train_df_original[numeric_cols], train_df_original[y_cols]
    test_df, y_test = test_df_original[numeric_cols], test_df_original[y_cols]
    y_train.drop(y_train.columns[2:], axis=1, inplace=True)
    y_test.drop(y_test.columns[2:], axis=1, inplace=True)

    combined = get_transfomed_combiner(train_df)

    x_train_transformed = combined.transform(train_df).astype(np.float32)
    x_test_transformed = combined.transform(test_df).astype(np.float32)

    return x_train_transformed, x_test_transformed, y_train, y_test


if __name__ == '__main__':

    ticker_dict = defaultdict(bool)

    my_list = list(my_list)
    for ticker in my_list:

        if ticker in ticker_dict:
            continue

        print('Processing: {}...'.format(ticker))

        try:
            data_list = get_input_target(ticker)
        except Exception as e:
            print(e)
            continue

        file_names = ('_x_train', '_x_test', '_y_train', '_y_test')

        if data_list[0] is not None:

            for file_name, data in zip(file_names, data_list):
                f_name = 'data/ohlc_processed/' + ticker + file_name + '.pickle'
                with open(f_name, 'wb') as handle:
                    pickle.dump(data, handle)

            ticker_dict[ticker] = True