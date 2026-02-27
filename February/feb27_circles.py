import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from Feb27_export import export_model

data = pd.read_csv("feb27_circles.csv")
X = torch.tensor(data.drop('y', axis=1).values).float()
Y = torch.tensor(data['y'].values).float().reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(2, 2), # 2 features, 2 output
    nn.ReLU(),
    nn.Linear(2, 1)
) # not including Sigmoid because it occurs in the loss function
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(loss)

export_model(model, 'feb27-model.json')