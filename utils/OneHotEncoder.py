import torch

class OneHotEncoder(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        target_onehot = torch.zeros(10)
        target_onehot[label] = 1.0
        return target_onehot