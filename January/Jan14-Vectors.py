import torch
from matrepr import mprint
import numpy as np
'''
X = torch.tensor([
    [1],
    [2],
    [3]
    ])

print(X)

print(X.dim())

print(X.shape)
'''
'''
# Printing matrix

Y = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
    ])

print(Y)

mprint(Y) # using matrepr to print matrix nicely

mprint(Y.T) # transpose of Y
'''