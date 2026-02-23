import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

torch.manual_seed(1) # for reproducibility

data = pd.read_csv('Feb20-carfuel.csv')
features = torch.tensor(data.drop('MPG', axis=1).to_numpy()).float()
target = torch.tensor(data['MPG'].to_numpy()).float().reshape(-1, 1)

fm = features.mean(axis = 0, keepdim = True)
fs = features.std(axis = 0, keepdim = True)
tm = target.mean(axis = 0, keepdim = True)
ts = target.std(axis = 0, keepdim = True)

X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(2, 1) # 2 inputs, 1 output
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs):
    Yhat = model(X) # forward pass, is the output
    loss = criterion(Yhat, Y) # calculating the loss, Y has to come last
    loss.backward() # backpropagation, calculating the gradients
    optimizer.step() # updating the weights and biases
    optimizer.zero_grad() 

torch.save({
    'fm': fm,
    'fs': fs,
    'tm': tm,
    'ts': ts,
    'parameters': model.state_dict()
}, 'feb20-model.pth')