from train_supervised_ohlc_per_day_cross_entropy import get_args_and_loggers, main
from itertools import product


def run_param_search():
    block_depths = [2, 3]
    const_factors = [8, 16]
    learning_rate = [0.01, 0.005]
    linear_dim = [0, 3]
    percentiles = [0.2, 0.3, 0.7, 0.8]

    for item in product(block_depths,
                        const_factors,
                        learning_rate,
                        linear_dim,
                        percentiles):
        block_depth, const_factor, lr, l_dim, percentile = item
        args, bce_logger, stat_logger = get_args_and_loggers()
        args.block_depth = block_depth
        args.const_factor = const_factor
        args.learning_rate = lr
        args.linear_dim = l_dim
        args.percentile = percentile
        main(args, bce_logger, stat_logger)


if __name__ == '__main__':
    run_param_search()
