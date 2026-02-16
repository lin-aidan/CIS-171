import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv("Feb16-data.csv")

features = torch.tensor(data.drop("Price", axis=1).to_numpy()).float()
target = torch.tensor(data['Price'].to_numpy()).float().reshape(-1, 1)

# Reshaping to 2 dimensional vectors
fm = features.mean().reshape(-1, 1)
fs = features.std().reshape(-1, 1)
tm = target.mean().reshape(-1, 1)
ts = target.std().reshape(-1, 1)

# Standardizing targets and features
X = (features - fm)/fs
Y = (target - tm)/ts

# Setting up neural network
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) # parameter refers to weights and biases

epochs = 100
for epoch in range(epochs):
    Yhat = model(X) # forward pass, is the output
    loss = criterion(Yhat, Y) # calculating the loss, Y has to come last
    loss.backward() # backpropagation, calculating the gradients
    optimizer.step() # updating the weights and biases
    optimizer.zero_grad() # resetting the gradients to zero for the next iteration


torch.save({
    'fm': fm,
    'fs': fs,
    'tm': tm,
    'ts': ts,
    'parameters': model.state_dict()
}, 'model.pth')