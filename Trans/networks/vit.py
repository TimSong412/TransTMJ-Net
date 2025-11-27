import torch
import torchvision
from PIL import Image
from accelerate import Accelerator
import numpy as np
import tqdm
import open_clip
import matplotlib.pyplot as plt
import os
from torch.utils.checkpoint import checkpoint
from transformers import ViTConfig, ViTModel



class ViT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.vitconfig = ViTConfig()
        self.model = ViTModel(self.vitconfig)