import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from Feb27_export import export_model

torch.manual_seed(42)

data = pd.read_csv("Feb27-lin-sep.csv")
X = torch.tensor(data.drop('y', axis=1).values).float()
Y = torch.tensor(data['y'].values).float().reshape(-1, 1)

model = nn.Linear(2, 1) # 2 features, 1 output
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