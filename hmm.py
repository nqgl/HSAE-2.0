import torch
import torch.nn as nn


class ActsTracker(nn.Module):
    def __init__(self):
        

    def forward(self, acts):
        self.update_acts(acts)
        return acts
