import torch

X_raw = torch.tensor([
    [1], 
    [5], 
    [9]
]).float()

Y_raw = torch.tensor([
    [5], 
    [8], 
    [2]
]).float()

Xm = X_raw.mean()
Xs = X_raw.std()

Ym = Y_raw.mean()
Ys = Y_raw.std()

X = (X_raw - Xm)/Xs

Y = (Y_raw - Ym)/Ys

w = torch.tensor([[0.0]], requires_grad=True)

b = torch.tensor([[0.0]], requires_grad=True)

epochs = 1000
lr = 0.01

for epoch in range(epochs):
    Yhat = X @ w + b
    r = Yhat - Y
    SSE = r.T @ r
    loss = SSE / X.shape[0]
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    print(loss.item(), w.item(), b.item())