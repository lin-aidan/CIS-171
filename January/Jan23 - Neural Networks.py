# Feed forward Neural Networks
import torch
import numpy as np
from matrepr import mprint

# X = torch.tensor(4)
X = torch.tensor([
    [4],
    [6]
    ])
w = torch.tensor(3)
b = torch.tensor(10)
Y = torch.tensor([
    [5], 
    [16]
])

Yhat = X*w + b
# loss = (Yhat - Y)**2
Residual = Yhat - Y
SSE = Residual.T @ Residual
loss = SSE/len(Residual)

# mprint(Yhat)
mprint(loss)