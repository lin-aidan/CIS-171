import torch
'''
X = torch.tensor([
    [2.0],
    [5.0]
])

Y = torch.tensor([
    [5.0],
    [1.0]
])

w = torch.tensor([
    [3.0]
])

b = torch.tensor([
    [1.0]
])
'''

X = torch.tensor([
    [2, 3]
]).float()

Y = torch.tensor([
    [30]
]).float()

w = torch.tensor([
    [4],
    [5]
]).float()

b = torch.tensor([
    [1]
]).float()

Yhat = X @ w + b
Residual = Yhat - Y 
SSE = Residual.T @ Residual
loss = SSE / 1
print(loss.item())