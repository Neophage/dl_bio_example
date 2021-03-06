from DLBio.pt_train_printer import IPrinterFcn
from DLBio.pt_training import ITrainInterface
import torch.nn as nn
import numpy as np
import torch


def get_interface(ti_type, model, device, printer, **kwargs):
    if ti_type == Classification.name:
        return Classification(model, device, printer)


class Classification(ITrainInterface):
    name = 'classification'

    def __init__(self, model, device, printer):
        self.printer = printer
        self.model = model
        self.xent_loss = nn.CrossEntropyLoss()
        self.functions = {
            'acc': Accuracy(),
            'er': ErrorRate()
        }
        self.d = device

    def train_step(self, sample):
        images, targets = sample[0].to(self.d), sample[1].to(self.d)
        pred = self.model(images)

        loss = self.xent_loss(pred, targets[:, 0])
        assert not bool(torch.isnan(loss))
        metrics = None
        counters = None
        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return loss, metrics, counters, functions


class Accuracy(IPrinterFcn):
    name = 'acc'

    def __init__(self):
        self.restart()

    def update(self, y_pred, y_gt):
        # get class with highest value
        y_pred = np.array(y_pred.detach().cpu())
        y_pred = np.argmax(y_pred, 1)
        self.x.append(y_pred)
        self.y.append(np.array(y_gt.detach().cpu()).flatten())
        return self

    def restart(self):
        self.name = type(self).name
        self.x = []
        self.y = []

    def __call__(self):
        x = np.concatenate(self.x)
        y = np.concatenate(self.y)
        return (x == y).mean()


class ErrorRate(Accuracy):
    name = 'e_rate'

    def __init__(self):
        self.restart()

    def __call__(self):
        acc = super(ErrorRate, self).__call__()
        return (1. - acc) * 100.
