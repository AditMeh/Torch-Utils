import copy
import torch


class StatsTracker():
    def __init__(self, mean_denom_train, mean_denom_val):
        self.train_hist = []
        self.val_hist = []
        self.train_loss_curr = 0.0
        self.val_loss_curr = 0.0
        # Denominator used to compute mean loss per epoch, batch_size * number of samples
        self.mean_denom_train = mean_denom_train
        self.mean_denom_val = mean_denom_val

        self.best_model = None
        self.best_val_loss_value = torch.inf

    def update_histories(self, train_value=None, val_value=None, net=None):
        if train_value is not None:
            self.train_hist.append(train_value)
        if val_value is not None:
            self.val_hist.append(val_value)
            if val_value < self.best_val_loss_value:
                self.best_val_loss_value = val_value
                self.store_model(net)

    def update_curr_losses(self, train_value=None, val_value=None):
        if train_value is not None:
            self.train_loss_curr += train_value
        if val_value is not None:
            self.val_loss_curr += val_value

    def compute_means(self):
        return (self.train_loss_curr/self.mean_denom_train), (self.val_loss_curr/self.mean_denom_val)

    def reset(self):
        self.train_loss_curr = 0.0
        self.val_loss_curr = 0.0

    def store_model(self, net):
        self.best_model = copy.deepcopy(net.state_dict())
