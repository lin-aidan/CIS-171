import torch
from matrepr import mprint
import numpy as np

A = torch.tensor([
    [1, 3],
    [5, 2]
])

B = torch.tensor([
    [2, 7],
    [10, 1]
])

# Addition with matrices

# mprint(A+B)


# Multiplication with matrices
'''Hadamard Product (element-wise multiplication)
- Multiply An by Bn to get Cn
- Cn = An * Bn'''

# mprint(A*B)

'''Scalar Multiplication'''
mprint(4*A)

