import torch
'''
X = torch.tensor([3.0])

Y = torch.tensor([10.0])

w = torch.tensor([6.0], requires_grad=True)

b = torch.tensor([1.0], requires_grad=True)
'''
X = torch.tensor([
    [1], 
    [5], 
    [8]
]).float()

Y = torch.tensor([3, 6, 1]).float()

w = torch.tensor([0.0], requires_grad=True)

b = torch.tensor([0.0], requires_grad=True)

Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE/X.shape[0]

lr = 0.2

loss.backward()

with torch.no_grad():
    w -= lr*w.grad
    b -= lr*b.grad

w.grad.zero_()
b.grad.zero_()

print(f"w = {w}")
print(f"b = {b}")