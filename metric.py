import numpy as np
import sklearn.metrics as metrics


class Metrics(object):
    def __init__(self, groundtruth: np.array, logits: np.array) -> None:
        self.gt = groundtruth
        self.logits = logits
        self.preds = logits > 0.5

    def metric_for_binary_classification(self):
        acc = metrics.accuracy_score(self.gt, self.preds)
        f1_macro = metrics.f1_score(self.gt, self.preds, average="macro")
        f1_micro = metrics.f1_score(self.gt, self.preds, average="micro")
        auc_roc = metrics.roc_auc_score(self.gt, self.logits)

        return acc, f1_macro, auc_roc
