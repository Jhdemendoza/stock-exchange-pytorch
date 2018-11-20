import argparse
import datetime
import logging
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import FeatureUnion

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ignite.metrics import BinaryAccuracy, Loss, Precision, Recall
from ignite.engine import Events, \
                          create_supervised_trainer, \
                          create_supervised_evaluator

import sklearn.metrics as sk_metrics
from supervised import load_dataframes, get_y_cols, TickerDataSimple
from functools import partial
from ignite.metrics import EpochMetric
from download_daily_data import my_list


class LogisticRegressor(nn.Module):
    def __init__(self, input_size, final_output_size):
        super(LogisticRegressor, self).__init__()
        self.e1 = nn.Embedding(5, 2)
        self.e2 = nn.Embedding(7, 2)
        # +2 for embedding... 2+2-2
        self.l1 = nn.Linear(input_size + 2, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, final_output_size)

    def forward(self, x):
        x1, x2, = x[:, -2].long(), x[:, -1].long()
        x1 = self.e1(x1)
        x2 = self.e2(x2)

        x = torch.cat([x[:, :-2], x1, x2], dim=1)
        x = torch.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        return torch.sigmoid(self.l4(x))


class CustomLoss(torch.nn.Module):
    '''
    Implement a simple verison of Focal Loss
    '''

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_target):
        y_pred = y_pred.flatten()
        y_target = y_target.flatten()

        def log_p(pred, target):
            return -((1 - pred) * torch.log2(pred) * target)

        return (log_p(y_pred,     y_target) +
                log_p(1 - y_pred, 1 - y_target)).mean()


def sk_metric_fn(y_pred, y_targets, sk_metrics, activation=None):
    y_true = y_targets.flatten().numpy()
    y_pred = y_pred.flatten().numpy()
    if activation is not None:
        y_pred = activation(y_pred)

    return sk_metrics(y_true, y_pred)


class ROC_AUC(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(ROC_AUC, self).__init__(
            partial(sk_metric_fn,
                    sk_metrics=sk_metrics.roc_auc_score,
                    activation=activation),
            output_transform=output_transform)


class F1_Score(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(F1_Score, self).__init__(
            partial(sk_metric_fn,
                    sk_metrics=sk_metrics.f1_score,
                    activation=activation),
            output_transform=output_transform)


class BinaryAccuracy(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(BinaryAccuracy, self).__init__(
            partial(sk_metric_fn,
                    sk_metrics=sk_metrics.accuracy_score,
                    activation=activation),
            output_transform=output_transform)


class Precision(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(Precision, self).__init__(
            partial(sk_metric_fn,
                    sk_metrics=sk_metrics.precision_score,
                    activation=activation),
            output_transform=output_transform)


class Recall(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(Recall, self).__init__(
            partial(sk_metric_fn,
                    sk_metrics=sk_metrics.recall_score,
                    activation=activation),
            output_transform=output_transform)


class ConfusionMatrix(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(ConfusionMatrix, self).__init__(
            partial(sk_metric_fn,
                    sk_metrics=sk_metrics.confusion_matrix,
                    activation=activation),
            output_transform=output_transform)


def zero_one(y_preds):
    return y_preds > 0.5


def zero_one_transform(output):
    return (zero_one(output[0])).long(), output[1].long()


def get_transfomed_combiner():
    # Use only the ones worked well in autoencoder
    transfomer = [
        ('Data after min-max scaling',
         MinMaxScaler()),
        ('Data after max-abs scaling',
         MaxAbsScaler()),
        ('Data after quantile transformation (uniform pdf)',
         QuantileTransformer(output_distribution='uniform')),
        ('Data after sample-wise L2 normalizing',
         Normalizer()),
    ]

    combined = FeatureUnion(transfomer)
    _ = combined.fit(train_df)

    return combined


def get_input_target(ticker):
    # messy code...
    train_df_original, test_df_original, numeric_cols, categoric_cols = load_dataframes(ticker)
    if train_df_original is None:
        return None, None, None, None

    y_cols, not_interested = get_y_cols(numeric_cols)
    numeric_cols = list(set(numeric_cols) - set(y_cols) - set(not_interested))
    # Let's not do month embedding
    categoric_cols.remove('month')
    categorical_train_embeddings = train_df_original[categoric_cols].applymap(lambda x: int(x))
    categorical_test_embeddings = test_df_original[categoric_cols].applymap(lambda x: int(x))

    train_df, y_train = train_df_original[numeric_cols], train_df_original[y_cols]
    test_df, y_test = test_df_original[numeric_cols], test_df_original[y_cols]
    y_train.drop(y_train.columns[4:], axis=1, inplace=True)
    y_test.drop(y_test.columns[4:], axis=1, inplace=True)
    binary_y_train = (y_train > args.threshold).astype(np.int)
    binary_y_test = (y_test > args.threshold).astype(np.int)

    combined = get_transfomed_combiner()

    x_train_transformed = combined.transform(train_df)
    x_test_transformed = combined.transform(test_df)
    x_train_numerical_transformed = torch.from_numpy(x_train_transformed).float()
    x_test_numerical_transformed = torch.from_numpy(x_test_transformed).float()

    x_train_categorical_cols = torch.from_numpy(categorical_train_embeddings.values).float()
    x_train_categorical_cols[:, 1] -= 9 # Should just be min of the column...
    x_test_categorical_cols = torch.from_numpy(categorical_test_embeddings.values).float()
    x_test_categorical_cols[:, 1] -= 9

    x_train_all = torch.cat([x_train_numerical_transformed, x_train_categorical_cols], dim=1)
    x_test_all = torch.cat([x_test_numerical_transformed, x_test_categorical_cols], dim=1)

    return x_train_all, binary_y_train, x_test_all, binary_y_test


def log_trainers(trainer):

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_dl)
        metrics = evaluator.state.metrics
        if trainer.state.epoch % args.print_every == 0:
            print("Training Results  - Epoch: {} Avg accuracy: {:.5f}, Avg BCE: {:.5f}, F1 Score: {:.5f}, ROC_AUC: {:.5f}".format(
                trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['f1_score'], metrics['roc_auc'],))
            print("Training Results  - Epoch: {} Precision: {:.5f}, Recall: {:.5f}".format(
                trainer.state.epoch, metrics['precision'], metrics['recall'],))
            print("Training Results  - Epoch: {} Confusion Matrix: \n{}".format(
                trainer.state.epoch, metrics['conf_matrix'], ))
            bce_logger.info("Training Results  - Epoch: {} Avg accuracy: {:.5f}, Avg BCE: {:.5f}, F1 Score: {:.5f}, ROC_AUC: {:.5f}".format(
                trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['f1_score'], metrics['roc_auc'],))
            bce_logger.info("Training Results  - Epoch: {} Precision: {:.5f}, Recall: {:.5f}".format(
                trainer.state.epoch, metrics['precision'], metrics['recall'],))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_dl)
        metrics = evaluator.state.metrics
        if trainer.state.epoch % args.print_every == 0:
            print("Validation Results- Epoch: {} Avg accuracy: {:.5f}, Avg BCE: {:.5f}, F1 Score: {:.5f}, ROC_AUC: {:.5f}".format(
                trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['f1_score'],metrics['roc_auc'],))
            print("Validation Results- Epoch: {} Precision: {:.5f}, Recall: {:.5f}".format(
                trainer.state.epoch, metrics['precision'], metrics['recall'],))
            print("Validation Results- Epoch: {} Confusion Matrix: \n{}".format(
                trainer.state.epoch, metrics['conf_matrix'], ))
            bce_logger.info("Validation Results- Epoch: {} Avg accuracy: {:.5f}, Avg BCE: {:.5f}, F1 Score: {:.5f}, ROC_AUC: {:.5f}".format(
                trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['f1_score'],metrics['roc_auc'],))
            bce_logger.info("Validation Results- Epoch: {} Precision: {:.5f}, Recall: {:.5f}".format(
                trainer.state.epoch, metrics['precision'], metrics['recall'],))


try:
    os.makedirs('logs')
except OSError:
    print('--- log folder exists')

FILE_NAME = 'logs/training_bce_{}.log'.format('_'.join(str(datetime.datetime.now()).split(' ')))

bce_logger = logging.getLogger(__name__)
bce_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(FILE_NAME)
file_handler.setFormatter(formatter)

parser = argparse.ArgumentParser(description='Hyper-parameters for the training')
parser.add_argument('--max_epoch',       default=30, type=int)
parser.add_argument('--print_every',     default=10, type=int)
parser.add_argument('--batch_size',      default=64, type=int)
parser.add_argument('--threshold',       default=0.0015, type=float)

args = parser.parse_args()
args.tickers = list(my_list)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


if __name__ == '__main__':

    for ticker in args.tickers:

        print('--- Starting training for {}'.format(ticker))

        x_train_all, binary_y_train, x_test_all, binary_y_test = get_input_target(ticker)
        if x_train_all is None: continue

        spy_dataset = TickerDataSimple(ticker, x_train_all, torch.from_numpy(binary_y_train.values).float())
        spy_testset = TickerDataSimple(ticker, x_test_all,  torch.from_numpy(binary_y_test.values).float())

        train_dl = DataLoader(spy_dataset, num_workers=1, batch_size=args.batch_size)
        test_dl = DataLoader(spy_testset, num_workers=1, batch_size=args.batch_size)

        # Transformer has 4 different ways
        model = LogisticRegressor(x_train_all.shape[1], binary_y_train.shape[1])
        # criterion = CustomLoss()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)

        trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
        evaluator = create_supervised_evaluator(
            model,
            metrics={
                'accuracy':     BinaryAccuracy(output_transform=zero_one_transform),
                'bce':          Loss(nn.modules.loss.BCELoss()),
                'f1_score':     F1_Score(output_transform=zero_one_transform),
                'roc_auc':      ROC_AUC(),
                'precision':    Precision(output_transform=zero_one_transform),
                'recall':       Recall(output_transform=zero_one_transform),
                'conf_matrix':  ConfusionMatrix(output_transform=zero_one_transform),},
            device=device)

        log_trainers(trainer)

        trainer.run(train_dl, max_epochs=args.max_epoch)
        print('--- Ending training for {}'.format(ticker))

        del trainer, evaluator, model, x_train_all, x_test_all
