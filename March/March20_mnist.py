import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from grid import save_image_grid

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
    batch_size = 10
)

for i, (images, labels) in enumerate(loader):
    # print(i, label)
    save_image_grid(images)
    if i == 9:
        break