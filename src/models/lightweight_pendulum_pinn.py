# Author: Berkan Mertan
# Copyright (c) 2025 Berkan Mertan. All rights reserved.
# This PINN will be used to learn the function theta(t) for pendulum systems
# With much less parameters, this is the LIGHTWEIGHT version of the regular pendulum pinn

import torch.nn as nn

class PendulumPINN(nn.Module):
    def __init__(self):
        super(PendulumPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)