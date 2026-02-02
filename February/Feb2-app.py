import torch

'''
X = torch.tensor(3.0, requires_grad=True)
f = X**2
f.backward() # backward for back propagation
print(X.grad)

# by not redefining X, the gradient accumulates (6 + 6)
X.grad.zero_()  # reset the gradient to zero
f = X**2
f.backward()
print(X.grad) # by not redefining X, the gradient accumulates (6 + 6)

X = torch.tensor(7.0, requires_grad=True)
f = (X**2 + 1)/(X + 5)
f.backward()
print(X.grad)
'''

X = torch.tensor(3.0, requires_grad=True)
Y = torch.tensor(-1.0, requires_grad=True)
Z = torch.tensor(0.0, requires_grad=True)

f = -3*Z**3*Y**3 + 3*X + 2*Z 
f.backward() # calculates 3 different derivatives with respect to X, Y, Z
print(X.grad, Y.grad, Z.grad)