import torch
from matrepr import mprint
import numpy as np

# Matrix Multiplication
'''
A = torch.tensor([
    [2, 7],
    [3, 4]
])

B = torch.tensor([
    [1, 2],
    [5, 3]
])
'''
A = torch.tensor([
    [0],
    [-5],
    [1]
])

B = torch.tensor([
    [-1, -4, -1, -1]
])

#print(A@B)
mprint(A@B)