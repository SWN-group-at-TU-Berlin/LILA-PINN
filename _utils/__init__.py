#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:52:36 2023

@author: ivodaniel
"""

import sys,warnings,pickle,string
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score,mean_squared_error
from itertools import permutations


##### Data utils #####

def get_loader(X, y=None, batch_size=1, shuffle=False):
    """Convert X and y Tensors to a DataLoader

        If y is None, use a dummy Tensor
    """
    if y is None:
        y = torch.Tensor(X.size()[0])
    return DataLoader(TensorDataset(X, y), batch_size, shuffle)


##### Logging #####

def add_metrics_to_log(log, metrics, y_true, y_pred, prefix=''):
    for metric in metrics:
        q = metric(y_true, y_pred)
        log[prefix + metric.__name__] = q
    return log


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "e}"
    return "    ".join(fmt.format(k, v) for k, v in log.items())


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n      = max(1, n)
        self.nf     = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i/100.0 * n) for i in range(101)])
        self.ticks.add(n-1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "="*b, " "*(self.length-b), int(100*((i+1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()

path = '../_utils/Data/leak_ground_truth/'
leak_signals = pd.read_csv(path+'2019_Leakages.csv',sep=';',decimal=',',parse_dates=['Timestamp'])
leak_signals.index = leak_signals['Timestamp'].values
leak_signals = leak_signals.drop(['Timestamp'],axis=1)
leak_signals /= 3600
