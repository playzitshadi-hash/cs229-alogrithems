import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2
class Embedding_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 16)
        self.embedding = nn.Linear(16, 8)
        # Use LeakyReLU to prevent dying neurons with small input spaces
        self.relu = nn.LeakyReLU(0.1) 
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        z = self.embedding(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

# Data setup
N = 300
generate0 = np.random.randn(300, 2) + np.array([0, 0])
generate1 = np.random.randn(300, 2) + np.array([4, 4])
generate2 = np.random.randn(300, 2) + np.array([0, 4])
X = np.vstack((generate0, generate1, generate2))
view1 = X + 0.3 * np.random.randn(900, 2)
view2 = X + 0.3* np.random.randn(900, 2)

model = Embedding_Extractor()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Slightly reduced LR

# Standard contrastive temperature
tau = 0.01 

view1_tensor = torch.FloatTensor(view1)
view2_tensor = torch.FloatTensor(view2)

print("Training with updated configuration...")
for epoch in range(500):
    optimizer.zero_grad()
    
    z1 = model(view1_tensor)
    z2 = model(view2_tensor)
    
    cosine_sim = z1 @ z2.T 
    logits = cosine_sim / tau
    
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    loss = (loss1 + loss2) / 2
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
n = 900 
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    embeddings = model(X_tensor).numpy()
mean = np.mean(embeddings, axis=0)
centered_X = embeddings - mean
cov = (centered_X.T @ centered_X) / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
sort = np.argsort(eigenvalues)[::-1]
pca_matrix = eigenvectors[:, sort[:2]]
z_2d = centered_X @ pca_matrix
labels = np.array([0]*300 + [1]*300 + [2]*300)
colors = ["red","green","black"]
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(z_2d[labels==i,0], z_2d[labels==i,1], color=colors[i], label=f"Class {i}", alpha=0.7)
plt.title("PCA of Contrastive Embeddings")
plt.legend()
plt.show()