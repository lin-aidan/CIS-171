import torch

Y = torch.tensor([
    [1],
    [5],
    [4]
])

Yhat = torch.tensor([
    [2],
    [4],
    [6]
])

r = Yhat-Y
SSE = r.T@r
print(SSE)
loss = SSE/Y.shape[0]
loss2 = SSE/3
print(loss, loss2)