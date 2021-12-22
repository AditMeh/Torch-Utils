import torch
import copy
import torch.nn.functional as F


class OneHotEncoder(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        target_onehot = torch.zeros(10)
        target_onehot[label] = 1.0
        return target_onehot




class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = torch.inf
        self.stop = False
        self.epoch_counts = 0
        self.best_model = None

    def __call__(self, val_loss):
        if self.best_val_loss > val_loss + self.delta:
            self.best_val_loss = val_loss
            self.epoch_counts = 0

        else:
            self.epoch_counts += 1

            if self.epoch_counts == self.patience:
                self.stop = True

