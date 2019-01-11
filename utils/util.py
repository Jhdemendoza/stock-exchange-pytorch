import datetime
import os
import re
import sys
import traceback
from functools import wraps
from ignite.engine import Events
from ignite.handlers import EarlyStopping


def create_path(args, folder_path, is_log_path=False):
    pattern = re.compile(r'[\w]+=[\w.\']+')
    to_add = (','.join(pattern.findall(args.__str__())) +
              datetime.datetime.today().date().__str__() +
              '/' if not is_log_path else '')
    args.folder_path = folder_path + to_add
    try:
        os.makedirs(args.folder_path, exist_ok=True)
    except OSError:
        traceback.print_exc()
        sys.exit(3)
    # Do this purposely so that we don't forget!
    return args.folder_path


def print_and_log(msg, logger):
    print(f'{msg}')
    logger.info(f'{msg}')


def register_early_stopping(evaluator_test, trainer, args):
    def score_function(engine):
        val_loss = engine.state.metrics['bce']
        return val_loss
    early_stopping_handler = EarlyStopping(patience=args.patience,
                                           score_function=score_function,
                                           trainer=trainer)
    evaluator_test.add_event_handler(Events.COMPLETED, early_stopping_handler)


def wrap_model_in_eval_mode(model):
    def _wrap_model_in_eval_mode(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            model.eval()
            func(*args, **kwargs)
            model.train()
        return _wrapper
    return _wrap_model_in_eval_mode
