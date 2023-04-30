import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SRDataset
from models import SRCNN, VDSR, EDSR


def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model_name = model.get_name()
    model = model.to(device)
    model.train()

    print(f"Training {model_name.upper()} model:")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if model_name == 'vdsr':
                targets = targets - inputs

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

    torch.save(model.state_dict(), f'../models/{model_name}.pt')


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

train_dataset = SRDataset('../dataset/train/lr', '../dataset/train/hr', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# model, criterion = SRCNN(), nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# train(model, device, train_loader, criterion, optimizer, num_epochs=100)

# model, criterion = VDSR(), nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# train(model, device, train_loader, criterion, optimizer, num_epochs=100)

model, criterion = EDSR(), nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train(model, device, train_loader, criterion, optimizer, num_epochs=100)
