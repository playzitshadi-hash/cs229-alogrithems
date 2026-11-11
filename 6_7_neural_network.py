import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim



X, y = make_moons(n_samples=500, noise=0.20, random_state=42)
y = y.reshape(-1, 1)  
m = len(X)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y).reshape(-1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], color="red", label="Class 0")
plt.scatter(
    X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], color="black", label="Class 1"
)
model = LogisticRegression()
model.fit(X, y.ravel())

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
X_range = np.linspace(x_min, x_max, 100)
y_range = np.linspace(y_min, y_max, 100)
xx, yy = np.meshgrid(X_range, y_range)
d = np.c_[xx.ravel(), yy.ravel()]


log_preds = model.predict(d)
plt.contourf(xx, yy, log_preds.reshape(xx.shape), alpha=0.1, cmap="coolwarm")


np.random.seed(42)
W1 = np.random.randn(2, 16) * 0.01
W2 = np.random.randn(16,1) * 0.01
b1 = np.zeros((1,16))
b2 = np.zeros((1, 1))

alpha = 0.1


for i in range(10000):

  Z1 = X @ W1 + b1
  A1 = np.maximum(0, Z1) 
  Z2 = A1 @ W2 + b2
  y_prediction = 1 / (1 + np.exp(-Z2))  


  p_clipped = np.clip(y_prediction, 1e-9, 1 - 1e-9)
  j = -(1 / m) * np.sum(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
  if i % 1000 == 0:
    accuracy = np.mean((y_prediction > 0.5).astype(int) == y)
    print(f"Iterations :{i} Loss = {j} Accuracy = {accuracy} ")

  dZ2 = y_prediction - y
  dw2 = (1 / m) * (A1.T @ dZ2)
  db2 = np.mean(dZ2, axis=0, keepdims=True)

  dA1 = dZ2 @ W2.T
  dZ1 = dA1 * (Z1 > 0) 
  dw1 = (1 / m) * (X.T @ dZ1)
  db1 = np.mean(dZ1, axis=0, keepdims=True)

  W1 -= alpha * dw1
  W2 -= alpha * dw2
  b1 -= alpha * db1
  b2 -= alpha * db2
Z1_grid = d @ W1 + b1
A1_grid = np.maximum(0, Z1_grid)
Z2_grid = A1_grid @ W2 + b2
y_pred_grid = 1 / (1 + np.exp(-Z2_grid))

classes_grid = (y_pred_grid > 0.5).astype(int)
classes_2d = classes_grid.reshape(xx.shape)
plt.contourf(xx, yy, classes_2d, alpha=0.2, cmap='coolwarm')
plt.title("Make Moons with Neural Network & Logistic Regression Boundary")
plt.legend()
plt.show()


model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
for i in range(10000):

    y_pred = model(X_torch)
    loss = criterion(y_pred, y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predicted_classes = (y_pred > 0.5).float()
    accuracy2 = (predicted_classes == y_torch).float().mean().item()
    if i % 1000 == 0:
       print(f"Iteration {i}, Loss: {loss.item()}, Accuracy: {accuracy2}")

d_tensor = torch.FloatTensor(d)
y_pred_grid = model(d_tensor).detach().numpy()
classes_grid = (y_pred_grid > 0.5).astype(int)
classes_2d_pytorch = classes_grid.reshape(xx.shape)


plt.contourf(xx, yy, classes_2d, alpha=0.2, cmap='coolwarm')
plt.show()

print(f" My Loss = {j} Accuracy = {accuracy} \n")
print(f" Torch Loss = {loss.item()} Accuracy = {accuracy2} ")

