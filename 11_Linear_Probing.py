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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


images, labels = next(iter(train_loader))
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()
plt.close(fig)  


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


all_train_embeddings = []
all_train_labels = []
cnn_model.eval()
with torch.no_grad():
    for inputs, labels in train_loader:
        _, z = cnn_model(inputs)
        all_train_embeddings.append(z)
        all_train_labels.append(labels)

train_embeddings = torch.cat(all_train_embeddings)
train_labels = torch.cat(all_train_labels)


all_test_embeddings = []
all_test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        _, z = cnn_model(inputs)
        all_test_embeddings.append(z)
        all_test_labels.append(labels)

test_embeddings = torch.cat(all_test_embeddings)
test_labels = torch.cat(all_test_labels)

# 3. Train the Linear Probe on Extracted Embeddings
linear_probe = nn.Linear(64, 10)
probe_optimizer = optim.Adam(linear_probe.parameters(), lr=0.01)

linear_probe.train()
for epoch in range(5):
    probe_optimizer.zero_grad()
    probe_out = linear_probe(train_embeddings)
    probe_loss = criterion(probe_out, train_labels)
    probe_loss.backward()
    probe_optimizer.step()

# 4. Evaluate Linear Probe
linear_probe.eval()
with torch.no_grad():
    probe_out = linear_probe(test_embeddings)
    preds = torch.argmax(probe_out, dim=1)
    accuracy = (preds == test_labels).float().mean().item()

# 5. Evaluate Original CNN
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        out, _ = cnn_model(images)
        preds = torch.argmax(out, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    cnn_accuracy = correct / total

print(f"Original CNN Accuracy: {cnn_accuracy * 100:.2f}%")
print(f"Linear Probe Accuracy: {accuracy * 100:.2f}%")



test_emb_np = test_embeddings.numpy()
test_labels_np = test_labels.numpy()
mean = np.mean(train_embeddings.numpy(), axis=0)
centered_test = test_emb_np - mean
cov = (centered_test.T @ centered_test) / (len(test_emb_np) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
sort = np.argsort(eigenvalues)[::-1]
pca_matrix = eigenvectors[:, sort[:2]]
z_2d = centered_test @ pca_matrix
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(10):
    class_points = z_2d[test_labels_np == i]
    plt.scatter(class_points[:, 0], class_points[:, 1], color=colors[i], label=class_names[i], alpha=0.5)
plt.title("PCA of Fashion-MNIST Embeddings")
plt.legend()
plt.show()