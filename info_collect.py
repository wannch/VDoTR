import time

import log as logger
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class SingleEpochAccumulator:
    def __init__(self, model_name, ds_name, epoch=0) -> None:
        self.mn = model_name
        self.dn = ds_name
        self.epoch = epoch

        self.logger = logger.Logger("@".join(model_name, ds_name))
        self.start_time = time.time()

        self.loss = []
        self.right_preds = 0
        self.total = 0

    def __str__(self) -> str:
        stats_info = f"Mean loss each batch: {self.mean_loss()}"
        duration_time = f"Time: {(time.time() - self.start_time)}"

        return f"{stats_info}  {duration_time}"

    def add(self, l, rpc, yc):
        self.loss += l
        self.right_preds += rpc
        self.total += yc

    def acc(self):
        return self.right_preds / self.total

    def mean_loss(self):
        return sum(self.loss) / len(self.loss)

    def to_log(self):
        self.logger.info(str(self))

class History:
    def __init__(self, name):
        self.name = name
        self.history = {}
        self.epoch = 0
        self.timer = time.time()

    def __call__(self, stats, epoch):
        self.epoch = epoch
        if epoch in self.history:
            self.history[epoch].append(stats)
        else:
            self.history[epoch] = [stats]

    def __str__(self):
        epoch = f"\nEpoch {self.epoch} "
        stats = ' - '.join([f"{res}" for res in self.current()])
        timer = f"Time: {(time.time() - self.timer)}"

        return f"{self.name}  {epoch}  {stats}  {timer}"

    def current(self):
        return self.history[self.epoch]

    def log(self):
        msg = f"Epoch: {self.epoch}  {' - '.join([f'{res}' for res in self.current()])}  Time: {(time.time() - self.timer)}"
        logger.log_info(self.name, msg)

class Metrics:
    def __init__(self, outs, labels):
        self.scores = outs
        self.labels = labels
        self.transform()
        print(self.predicts)

    def transform(self):
        self.series = pd.Series(self.scores)
        self.predicts = self.series.apply(lambda x: 1 if x >= 0.5 else 0)
        self.predicts.reset_index(drop=True, inplace=True)

    def __str__(self):
        confusion = confusion_matrix(y_true=self.labels, y_pred=self.predicts)
        tn, fp, fn, tp = confusion.ravel()
        string = f"\nConfusion matrix: \n"
        string += f"{confusion}\n"
        string += f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        string += '\n'.join([name + ": " + str(metric) for name, metric in self().items()])
        return string

    def __call__(self):
        _metrics = {"Accuracy": metrics.accuracy_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision": metrics.precision_score(y_true=self.labels, y_pred=self.predicts),
                    "Recall": metrics.recall_score(y_true=self.labels, y_pred=self.predicts),
                    "F-measure": metrics.f1_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision-Recall AUC": metrics.average_precision_score(y_true=self.labels, y_score=self.scores),
                    "AUC": metrics.roc_auc_score(y_true=self.labels, y_score=self.scores),
                    "MCC": metrics.matthews_corrcoef(y_true=self.labels, y_pred=self.predicts)}
                    #,"Error": self.error()
        return _metrics

    def log(self):
        excluded = ["Precision-Recall AUC", "AUC"]
        _metrics = self()
        msg = ' - '.join(
            [f"({name[:3]} {round(metric, 3)})" for name, metric in _metrics.items() if name not in excluded])

        logger.log_info('metrics', msg)

    def error(self):
        errors = [(abs(score - (1 if score >= 0.5 else 0))/score)*100 for score, label in zip(self.scores, self.labels)]

        return sum(errors)/len(errors)