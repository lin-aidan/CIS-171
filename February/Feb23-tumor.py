import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

data = pd.read_csv("Feb23-tumor.csv")
data['Diagnosis'] = data['Diagnosis'].map({'Malignant': 1, 'Benign': 0})
features = torch.tensor(data.drop('Diagnosis', axis=1).to_numpy()).float()
target = torch.tensor(data['Diagnosis'].to_numpy()).float().reshape(-1, 1)

# Standardizing features
fm = features.mean(axis = 0, keepdim = True)
fs = features.std(axis = 0, keepdim = True)

# Not standardizing target since loss needs to know the actual values of 0 and 1 for binary classification

X = (features - fm)/fs
Y = target

model = nn.Linear(1, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 250

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


torch.save({
    'fm': fm,
    'fs': fs,
    'parameters': model.state_dict()
}, "Feb23-tumor.pth")