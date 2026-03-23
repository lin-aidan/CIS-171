import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from grid import save_image_grid

torch.manual_seed(1)

dataset = datasets.MNIST(
    root = './data', # creating another folder called data to store the MNIST dataset
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

# print(len(dataset))
# image, label = dataset[0]
# image.save('image.png')

loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = True
)
'''
for i, (images, labels) in enumerate(loader):
    # print(i, label)
    save_image_grid(images)
    if i == 9:
        break
'''

model =nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128), # 784 for the 28x28 pixels
    nn.ReLU(),
    nn.Linear(128,64), # 128 for the 128 neurons in the previous layer
    nn.ReLU(),
    nn.Linear(64,10) # 10 for the 10 classes in the MNIST dataset (digits 0-9)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    for X, Y in loader: # X,Y are done in batches of 64, not the whole set
        optimizer.zero_grad()
        Yhat = model(X)
        loss = criterion(Yhat, Y)
        loss.backward()
        optimizer.step()
print(loss)