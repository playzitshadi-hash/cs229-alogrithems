import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(2, 2)
        self.embedding = nn.Linear(1568, 64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1) 
        z = self.embedding(x)
        out = self.classifier(z)
        return out, z

# Dataset and Dataloaders
train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class LoRa_Linear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_linear
        # Freeze the original layer weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        
        # LoRA parameters
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = (x @ self.A @ self.B) * self.scaling
        return original_out + lora_out

cnn_model = FashionCNN()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5): 
    for images, labels in train_loader:
        out, z = cnn_model(images)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"CNN Epoch {epoch}, Loss: {loss.item():.4f}")

for param in cnn_model.parameters():
    param.requires_grad = False

cnn_model.classifier = LoRa_Linear(cnn_model.classifier, rank=4, alpha=1.0)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, cnn_model.parameters()), lr=0.001)

for epoch in range(3):
    cnn_model.train()
    for images, labels in train_loader:
        out, z = cnn_model(images)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"LoRA Epoch {epoch}, Loss: {loss.item():.4f}")

cnn_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        out, z = cnn_model(images)
        preds = torch.argmax(out, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    cnn_accuracy = correct / total

print(f"LoRA Fine-tuned Accuracy: {cnn_accuracy * 100:.2f}%")