import torch

X = torch.tensor([
    [1], 
    [5], 
    [9]
]).float()

Y = torch.tensor([
    [5], 
    [8], 
    [2]
]).float()

w = torch.tensor([[0.0]], requires_grad=True)

b = torch.tensor([[0.0]], requires_grad=True)

epochs = 500000
lr = 0.01

for i in range(epochs):
    Yhat = X@w + b
    r = Yhat - Y
    SSE = r.T@r
    loss = SSE/X.shape[0]
    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    #print(loss.item(), w, b)
    w.grad.zero_()
    b.grad.zero_()

print(7*w + b)