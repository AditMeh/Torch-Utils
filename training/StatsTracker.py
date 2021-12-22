import copy
import torch


class StatsTracker():
    def __init__(self):
        self.train_hist = []
        self.val_hist = []
        self.train_loss_curr = 0.0
        self.val_loss_curr = 0.0

        self.best_model = None
        self.best_val_loss_value = torch.inf

    def update_histories(self, train_value=None, val_value=None, net=None):
        if train_value is not None:
            self.train_hist.append(train_value)
        if val_value is not None:
            self.val_hist.append(val_value)
            if val_value < self.best_val_loss_value:
                print("save")
                
                self.best_val_loss_value = val_value
                self.store_model(net)

    def update_curr_losses(self, train_value=None, val_value=None):
        if train_value is not None:
            self.train_loss_curr += train_value
        if val_value is not None:
            self.val_loss_curr += val_value

    def reset(self):
        self.train_loss_curr = 0.0
        self.val_loss_curr = 0.0

    def store_model(self, net):
        self.best_model = copy.deepcopy(net.state_dict())
