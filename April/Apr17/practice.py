import torch
import torch.nn as nn
import numpy as np

embedding = nn.Embedding(27, 2)

print(embedding(torch.tensor(8)))