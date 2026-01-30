import torch
import pandas as pd
import numpy as np

data = pd.read_csv('Jan30-data.csv')

X = torch.tensor(data.drop('Y', axis=1).to_numpy()).float()

Y = torch.tensor(data['Y'].to_numpy()).float().reshape(-1, 1)

w = torch.tensor([
    [2], 
    [3], 
    [-2], 
    [2]
]).float()

b = torch.tensor([
    [3]
])

Yhat = X@w + b
Residual = Yhat - Y
SSE = Residual.T @ Residual
loss = SSE / X.shape[0]
print(loss)