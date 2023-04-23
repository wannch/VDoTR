import dataclasses
from dataclasses import dataclass
from typing import List

class Stat:
    def __init__(self, outs=None, labels=None, loss=0.0, right_preds=0, total_samples=0):
        if labels is None:
            labels = []
        if outs is None:
            outs = []

        self.outs = outs
        self.labels = labels
        self.loss = loss

        self.right_preds = right_preds
        self.total_samples = total_samples
        self.acc = 0.0

    def __add__(self, other):
        return Stat(self.outs + other.outs, self.labels + other.labels, self.loss + other.loss, self.right_preds + other.right_preds, self.total_samples + other.total_samples)

    def __str__(self):
        return f"Mean loss each batch: {round(self.loss, 4)}, Acc: {round(self.acc, 4)};"


@dataclass
class Stats:
    name: str
    results: List[Stat] = dataclasses.field(default_factory=list)
    total: Stat = Stat()

    def __call__(self, stat):
        self.total += stat
        self.results.append(stat)

    def __str__(self):
        return f"{self.name} {self.cal()}"

    def __len__(self):
        return len(self.results)

    def cal(self):
        res = Stat()
        res += self.total
        res.loss /= len(self)
        # res.acc /= len(self)
        res.acc = res.right_preds / res.total_samples
        return res

    def loss(self):
        return self.cal().loss

    def acc(self):
        return self.cal().acc

    def outs(self):
        return self.total.outs

    def labels(self):
        return self.total.labels
