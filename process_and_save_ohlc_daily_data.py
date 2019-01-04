import pickle
import numpy as np
from collections import defaultdict
from supervised import ohlc_get_y_cols, ohlc_train_df_test_df
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, QuantileTransformer
from sklearn.pipeline import FeatureUnion
import argparse


def get_args():
    # transform_dim should be 1 if transform==False... Identity Transform...
    parser = argparse.ArgumentParser(description='Hyper-parameters for the training')
    parser.add_argument('--data_point_dim',  default=5,     type=int)
    parser.add_argument('--transform',       default=True, type=bool)
    parser.add_argument('--transform_dim',   default=2,     type=int)
    parser.add_argument('--target_shift',           default=10,    type=int)
    parser.add_argument('--min_shift_forward',      default=4,     type=int)
    parser.add_argument('--max_shift_forward',      default=20,    type=int)
    parser.add_argument('--shift_increment',        default=5,     type=int)
    parser.add_argument('--create_more_features',   default=True,  type=bool)
    parser.add_argument('--folder_path',     default='data/ohlc_processed_transform/',  type=str)
    return parser.parse_args()


def get_transfomed_combiner(df):
    # Use only the ones worked well in autoencoder
    transformer = [
        ('Data after min-max scaling',
         MinMaxScaler()),
        # ('Data after max-abs scaling',
        #  MaxAbsScaler()),
        # ('Data after quantile transformation (uniform pdf)',
        #  QuantileTransformer(output_distribution='uniform')),
        ('Data after sample-wise L2 normalizing',
         Normalizer()),
    ]

    combined = FeatureUnion(transformer)
    _ = combined.fit(df)

    return combined


def get_input_target(ticker, args=None, transform=None):
    # messy code...
    train_df_original, test_df_original, numeric_cols, categoric_cols = ohlc_train_df_test_df(ticker, args=args)
    if train_df_original is None:
        return None, None, None, None

    y_cols, not_interested = ohlc_get_y_cols(numeric_cols)
    numeric_cols = list(sorted(set(numeric_cols) - set(y_cols) - set(not_interested)))

    train_df, y_train = train_df_original[numeric_cols].copy(), train_df_original[y_cols].copy()
    test_df, y_test = test_df_original[numeric_cols].copy(), test_df_original[y_cols].copy()
    y_train.drop(y_train.columns[2:], axis=1, inplace=True)
    y_test.drop(y_test.columns[2:], axis=1, inplace=True)

    if args.transform:
        combined = transform(train_df)
        x_train_transformed = combined.transform(train_df).astype(np.float32)
        x_test_transformed = combined.transform(test_df).astype(np.float32)
    else:
        x_train_transformed = train_df.astype(np.float32)
        x_test_transformed = test_df.astype(np.float32)

    return x_train_transformed, x_test_transformed, y_train, y_test


if __name__ == '__main__':

    args = get_args()

    from download_daily_data import all_tickers
    ticker_dict = defaultdict(bool)
    my_list = list(all_tickers)
    for ticker in my_list:

        # This is unncessary in the current setups..
        #     but if my_list has duplicates...
        if ticker in ticker_dict:
            continue

        print('Processing: {}...'.format(ticker))

        try:
            data_list = get_input_target(ticker,
                                         args=args,
                                         transform=get_transfomed_combiner)
        except Exception as e:
            print(e)
            continue

        file_names = ('_x_train', '_x_test', '_y_train', '_y_test')

        if data_list[0] is not None:
            # Eventually, make it so that it denotes data_point_dim and transform_dim
            transform_dim = args.transform_dim if args.transform else 1
            row_shape = data_list[0].shape[1]

            for file_name, data in zip(file_names, data_list):
                f_name = (args.folder_path
                          + ticker.lower()
                          + file_name
                          + '_transform_dim_{}'.format(transform_dim)
                          + '_data_shape_{}'.format(str(row_shape))
                          + '.pickle')
                with open(f_name, 'wb') as handle:
                    pickle.dump(data, handle)

            ticker_dict[ticker] = True