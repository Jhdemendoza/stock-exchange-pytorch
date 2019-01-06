from train_supervised_ohlc_per_day_cross_entropy_newly_processed import main, get_args
from itertools import product


def run_param_search():
    block_depths = [2, 3, 4]
    const_factors = [2, 4, 6]
    learning_rate = [0.007, ]
    linear_dim = [2, 4, 6]
    percentiles = [0.2, 0.25, 0.75, 0.8, ]

    for item in product(block_depths,
                        const_factors,
                        learning_rate,
                        linear_dim,
                        percentiles):
        block_depth, const_factor, lr, l_dim, percentile = item
        args = get_args()
        if percentile <= 0.2:
            args.max_epoch = 42
        elif percentile >= 0.8:
            args.max_epoch = 25
        args.block_depth = block_depth
        args.const_factor = const_factor
        args.learning_rate = lr
        args.linear_dim = l_dim
        args.percentile = percentile
        main(args)


if __name__ == '__main__':
    run_param_search()
