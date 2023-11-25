import os
import torch
import cv2
from torch import nn
import numpy as np

# x = torch.ones(1, dtype=torch.float32)
x = torch.Tensor([1, 2])
print(x.shape)
w1 = torch.ones((3, 3), dtype=torch.float32)
w2 = torch.ones((3, 3), dtype=torch.float32)

out = w1 * x[0] + w2 * x[1]
print(out)
