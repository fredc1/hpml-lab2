import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from resnet import ResNet, BasicBlock
from utils import progress_bar

# Set up argument parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--data', default='./data', type=str, help='Path to CIFAR10 dataset')
parser.add_argument('--workers', default=2, type=int, help='Number of dataloader workers')
parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer type (e.g., "sgd")')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Device to run on ("cuda" or "cpu")')
args = parser.parse_args()

device = args.device

# Prepare data with transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=args.workers)

# Build ResNet-18 model
model = ResNet(BasicBlock, [2, 2, 2, 2])
model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

# Training function
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Train for 5 epochs
for epoch in range(5):
    print("Starting Training\n")
    train(epoch)

