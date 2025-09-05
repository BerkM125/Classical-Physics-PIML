# Author: Berkan Mertan
# Copyright (c) 2025 Berkan Mertan. All rights reserved.
# Loads models from "out" folder if preferred
import torch
from models.pendulum_pinn import PendulumPINN

class ModelLoader():
    @staticmethod
    def load_model(path:str=None):
        """
        Loads a pre-trained model from a file.
        """
        model = PendulumPINN()
        model.load_state_dict(torch.load(path))
    
        model.eval()
        return model

    @staticmethod
    def save_model(model:PendulumPINN=None, path:str=None):
        """
        Saves a trained model to a file.
        """     
        torch.save(model.state_dict(), path)
   