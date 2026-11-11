import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
generate0= np.random.randn(300,2)
generate1= np.random.randn(300,2)
generate2= np.random.randn(300,2)
center_shift0= np.array([0,0])
center_shift1= np.array([3,3])
center_shift2=np.array([0,4])
class0 =  generate0+center_shift0
class1 =  generate1+center_shift1
class2 =  generate2+center_shift2




X= np.vstack((class0,class1,class2))
print(X.shape)

y_class0=np.zeros(300)
y_class1 = np.ones(300)
y_class2 = np.ones(300) *2
y = np.concatenate([y_class0,y_class1,y_class2])
print(X.shape)
print(y.shape)
X_torch = torch.FloatTensor(X)
y_torch = torch.LongTensor(y)
class Embedding_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.embedding = nn.Linear(16,8)
        self.classifier = nn.Linear(8,3)
    def forward(self,x):
        x = self.relu(self.hidden(x))
        z = self.embedding(x)
        out = self.classifier(z)
        return out,z
model = Embedding_Extractor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
for i in range(10000):

    y_pred,z = model(X_torch)
    loss = criterion(y_pred, y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predicted_classes = torch.argmax(y_pred ,dim=1)
    accuracy2 = (predicted_classes == y_torch).float().mean().item()
    if i % 1000 == 0:
       print(f"Iteration {i}, Loss: {loss.item()}, Accuracy: {accuracy2}")
with torch.no_grad():
    _, z = model(X_torch)
    z_np=z.numpy() 
    mean = np.mean(z_np,axis=0)
    n = len(z_np)
    print(mean.shape)
    centered_z = z_np - mean
    cov = (centered_z.T @ centered_z) / (n - 1)
    print(cov.shape)
    eigenvalues,eigenvectors = np.linalg.eigh(cov)
    sort = np.argsort(eigenvalues)[::-1]
    pca_matrix = eigenvectors[: ,sort[:2]]
    z_2d = centered_z@pca_matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    classes = [class0,class1,class2]
    colors = ["red","blue","black"]
    for i in range(10):
        column0=X[y == i, 0] 
        column1=X[y == i, 1]
        z_2dx = z_2d[y == i, 0]
        z_2dy= z_2d[y == i, 1]
        axes[0].scatter(column0,column1, color=colors[i], label=f"Class {i}")
        axes[1].scatter(z_2dx,z_2dy, color=colors[i], label=f"Class {i}")
    axes[0].set_title("Original Data (2D)")
    axes[1].set_title("Learned Embeddings (PCA of 8D)")
    axes[0].legend()
    axes[1].legend()
    plt.show()

