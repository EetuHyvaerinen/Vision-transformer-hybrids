import torch
from torchvision import datasets, transforms
import os
from config import *

train_transform = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.RandomCrop(IMAGE_SIZE, padding=2), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train = datasets.FashionMNIST(os.path.join(DATA_PATH, DATASET), train=True, download=True, transform=train_transform)

test_transform = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]), 
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
])
test = datasets.FashionMNIST(os.path.join(DATA_PATH, DATASET), train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_WORKERS,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=N_WORKERS,
    drop_last=False
)