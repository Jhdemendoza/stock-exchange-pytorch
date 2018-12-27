from functools import partial
from ignite.metrics import BinaryAccuracy, EpochMetric, Loss, Precision, Recall
from ignite._utils import convert_tensor

import numpy as np
import sklearn.metrics as sk_metrics

import torch
import torch.nn as nn


def sk_metric_fn(y_pred, y_targets, sk_metrics, activation=None):
    y_true = y_targets.flatten().numpy()
    y_pred = y_pred.flatten().numpy()
    if activation is not None:
        y_pred = activation(y_pred)
    return sk_metrics(y_true, y_pred)


class ROC_AUC(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(ROC_AUC, self).__init__(
            partial(sk_metric_fn, sk_metrics=sk_metrics.roc_auc_score, activation=activation),
            output_transform=output_transform)


class F1_Score(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(F1_Score, self).__init__(
            partial(sk_metric_fn, sk_metrics=sk_metrics.f1_score, activation=activation),
            output_transform=output_transform)


class BinaryAccuracy(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(BinaryAccuracy, self).__init__(
            partial(sk_metric_fn, sk_metrics=sk_metrics.accuracy_score, activation=activation),
            output_transform=output_transform)


class Precision(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(Precision, self).__init__(
            partial(sk_metric_fn, sk_metrics=sk_metrics.precision_score, activation=activation),
            output_transform=output_transform)


class Recall(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(Recall, self).__init__(
            partial(sk_metric_fn, sk_metrics=sk_metrics.recall_score, activation=activation),
            output_transform=output_transform)


class ConfusionMatrix(EpochMetric):
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(ConfusionMatrix, self).__init__(
            partial(sk_metric_fn, sk_metrics=sk_metrics.confusion_matrix, activation=activation),
            output_transform=output_transform)


class PositiveStatistics(EpochMetric):
    def __init__(self, non_binary_y, output_transform=lambda x: x, pred_threshold=0.5):
        super(PositiveStatistics, self).__init__(
            self.compute_stats, output_transform=output_transform)
        assert torch.is_tensor(non_binary_y), '::RAISE:: non_binary_y must be of type torch.tensor'
        self.non_binary_y = non_binary_y
        self.pred_threshold = pred_threshold

    def compute_stats(self, pred, target):

        assert self.non_binary_y.shape == pred.shape, \
            '::RAISE:: y.shape: {}, pred.shape: {}'.format(self.non_binary_y.shape, pred.shape)
        assert pred.shape == target.shape, \
            '::RAISE::: {} vs {}'.format(pred.shape, target.shape)

        mask = pred.ge(self.pred_threshold)
        relevant_pred = torch.masked_select(pred, mask)

        if relevant_pred.nelement() == 0:
            return 0.0, -1.0

        y_value = torch.masked_select(self.non_binary_y, mask)
        print('y_value.shape: {}, \n{}'.format(y_value.shape, y_value))
        distribution = relevant_pred * y_value

        assert distribution.shape == target.shape, \
            '::RAISE:::: {} vs {}'.format(distribution.shape, target.shape)

        return distribution.mean(), distribution.std()


def zero_one(y_preds):
    return y_preds > 0.5


def zero_one_transform(output):
    return (zero_one(output[0])).long(), output[1].long()


def get_metrics(non_binary_y_target):
    metrics = {'accuracy':         BinaryAccuracy(output_transform=zero_one_transform),
               'bce':              Loss(nn.modules.loss.BCELoss()),
               'f1_score':         F1_Score(output_transform=zero_one_transform),
               'roc_auc':          ROC_AUC(),
               'precision':        Precision(output_transform=zero_one_transform),
               'recall':           Recall(output_transform=zero_one_transform),
               'conf_matrix':      ConfusionMatrix(output_transform=zero_one_transform),
               # 'positive_stat':    PositiveStatistics(non_binary_y_target),
    }
    return metrics


def prepare_batch_empty_label(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options
    """
    x, _, y_transformed = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y_transformed, device=device, non_blocking=non_blocking))


def prepare_batch_all(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options
    """
    x, y, y_transformed = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
            convert_tensor(y_transformed, device=device, non_blocking=non_blocking))


def get_binary_target(non_binary_y, threshold, args):
    threshold_expanded = np.tile(threshold, [len(non_binary_y), 1])
    temp_result = non_binary_y >= threshold_expanded
    return temp_result if args.percentile >= 0.5 else ~temp_result
