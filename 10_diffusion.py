import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
batch_size = 32
T = 100

beta = np.linspace(0.00001, 0.02, T)
alpha = 1 - beta
at_hat = np.cumprod(alpha)

# Convert schedules to PyTorch tensors
alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
alpha_bar_tensor = torch.tensor(at_hat, dtype=torch.float32)
beta_tensor = torch.tensor(beta, dtype=torch.float32)

if 'image' not in globals():
    image = np.random.rand(16, 16)

def generate_augmented_batch(base_img, num_samples):
    augmented_batch = []
    base_pil = Image.fromarray((base_img * 255).astype(np.uint8))
    
    for _ in range(num_samples):
        angle = np.random.uniform(-30, 30)
        img_transformed = base_pil.rotate(angle, resample=Image.BILINEAR)
        
        # FIX: Use np.float32 instead of torch.float32 here
        arr = np.array(img_transformed, dtype=np.float32) / 255.0
        
        noise = np.random.normal(0, 0.03, arr.shape)
        arr = np.clip(arr + noise, 0, 1)
        augmented_batch.append(arr)
        
    return np.array(augmented_batch)

# Generate initial batch of augmented training images
aug_images_np = generate_augmented_batch(image, num_samples=batch_size)
x0 = torch.tensor(aug_images_np, dtype=torch.float32).unsqueeze(1) # Shape: (batch_size, 1, 16, 16)

# --- 3. Model Definition ---
class SimpleDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16)
        )
    def forward(self, x, t):
        x_flat = torch.flatten(x, start_dim=1)
        t_reshaped = t.view(-1, 1).float()  
        combined = torch.cat((x_flat, t_reshaped), dim=1)  
        return self.network(combined).view(-1, 1, 16, 16) 

model = SimpleDiffusionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. Training Loop with Dynamic Augmentation Refresh ---
for step in range(1000):
    # Refresh augmented training samples every 200 steps for dataset diversity
    if step % 200 == 0 and step > 0:
        fresh_aug = generate_augmented_batch(image, num_samples=batch_size)
        x0 = torch.tensor(fresh_aug, dtype=torch.float32).unsqueeze(1)

    t = torch.randint(0, T, (batch_size, 1), dtype=torch.long)
    epsilon = torch.randn_like(x0)
    t_normalized = t.float() / (T - 1)
    
    alpha_bar_t = alpha_bar_tensor[t].view(batch_size, 1, 1, 1)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
    
    predicted_epsilon = model(xt, t.float())
    if predicted_epsilon.shape != epsilon.shape:
        predicted_epsilon = predicted_epsilon.view_as(epsilon)
        
    loss = criterion(predicted_epsilon, epsilon)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# --- 5. Sampling / Reverse Diffusion Loop ---
model.eval()
xt = torch.randn((1, 1, 16, 16)) # Start from pure noise
saved_images = []

with torch.no_grad():
    for i in reversed(range(T)):
        t_tensor = torch.full((1, 1), i, dtype=torch.long)
        predicted_epsilon = model(xt, t_tensor.float())
        
        a_t = alpha_tensor[i]
        at_h = alpha_bar_tensor[i]
        b_t = beta_tensor[i]
        
        alpha_division = 1.0 / torch.sqrt(a_t)
        alpha_1 = torch.sqrt(1.0 - at_h)
        b_t = beta_tensor[i]
        alpha_bar_previous = alpha_bar_tensor[i-1]
        mean_xt_minus_1 = alpha_division * (xt - (b_t / alpha_1) * predicted_epsilon)
        posterior_varience = b_t *  (1- alpha_bar_previous) / (1.0 - alpha_bar_tensor[i]) 
        if i > 0:
            z = torch.randn_like(xt)
            sigma = torch.sqrt(torch.clamp(posterior_varience, min=1e-20))
            xt = mean_xt_minus_1 + sigma * z
        else:
            xt = mean_xt_minus_1
            
        if i in [75, 50, 25, 0]:
            saved_images.append(xt.clone())

# --- 6. Plotting Results ---
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
steps_to_show = [75, 50, 25, 0]

for idx, img in enumerate(saved_images):
    img_np = img.squeeze().cpu().numpy()
    axes[idx].imshow(img_np, cmap='gray')
    axes[idx].set_title(f"Step {steps_to_show[idx]}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

np.random.seed(42)

n = 50000

x1 = np.random.normal(0, 3, n)
x2 = np.random.normal(0, 2, n)




X_base = np.column_stack((x1, x2))
a=np.random.randn(2, 10)
noise = np.random.normal(0, 0.1, size=(n, 10))
X= (X_base @ a) + noise
print(X.shape)
mean = np.mean(X, axis=0)

centered_X = X - np.mean(X, axis=0)
cov = (centered_X.T @ centered_X) / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

sort = np.argsort(eigenvalues)[::-1]
print("Sorted indices:", sort)


pc1 = eigenvectors[:, sort[0]]
pc_x = [mean[0] - 3 * pc1[0], mean[0] + 3 * pc1[0]]
pc_y = [mean[1] - 3 * pc1[1], mean[1] + 3 * pc1[1]] 

top_2_vectors = eigenvectors[:, sort[:2]]


X_2D = centered_X @ top_2_vectors


plt.scatter(X_2D[:, 0], X_2D[:, 1], alpha=0.5)
plt.title("Data Projected onto Top 2 Principal Components")
plt.show()


u = eigenvectors[:, sort[0]]
a = np.dot(centered_X, u)
print("Shape of projected data (a):", a.shape)

explained_variance_ratio = eigenvalues[sort] / np.sum(eigenvalues)
# plt.plot(range(1, 11), explained_variance_ratio, marker='o')
# plt.title("Scree Plot")
# plt.xlabel("Principal Component")
# plt.ylabel("Variance Explained Ratio")
# plt.show()


print("First 5 projections:", a[:5])
# plt.hist(a, bins=30)
# plt.title("Histogram of Projections on PC1")
# plt.show()


np.random.seed(42)

mean0 = [2, 2]
mean1 = [5, 5]

cov0 = [[1.0, 0.5],
        [0.5, 1.0]]

cov1 = [[1.0, -0.3],
        [-0.3, 1.0]]

X0 = np.random.multivariate_normal(mean0, cov0, size=150)
X1 = np.random.multivariate_normal(mean1, cov1, size=150)
true_labels = np.concatenate([
    np.zeros(len(X0), dtype=int),
    np.ones(len(X1), dtype=int)
])
old_mean0 = mean0.copy()
old_mean1 = mean1.copy()
X = np.vstack((X0, X1))

# plt.scatter(X[:, 0], X[:, 1])
# plt.title("Unlabeled dataset")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

new_means=[]
K = 2
random_indices = np.random.choice(len(X), K, replace=False)
centroids = X[random_indices]
for j in range(2): 

  cluster_mean = ((centroids== j)[:, np.newaxis] * X).sum(axis=0) / (centroids == j).sum()
  new_means.append(cluster_mean)
mean0 = new_means[0]
mean1 = new_means[1]

new_centroids = []
e = 1e-4
max_iteration =100
for i in range(max_iteration):
        old_centroids = centroids.copy()
        print(centroids.shape)
        distances = ((X[:, np.newaxis] - centroids)**2).sum(axis=2)
        assignments = np.argmin(distances, axis=1)
        new_centroids = []
        for j in range(K):
          point_clusters = X[assignments == j]
        
          if len(point_clusters) > 0:

            cluster_mean = point_clusters.mean(axis=0)
            new_centroids.append(cluster_mean)
          else:

            random_point = X[np.random.choice(len(X))]
            new_centroids.append(random_point)
        centroids = np.array(new_centroids)
        dist_moved = np.linalg.norm(centroids[0] - old_centroids[0]) + np.linalg.norm(centroids[1] - old_centroids[1])
        if dist_moved < e:
         print(f"Converged at iteration {i}")
         break
colors = ['red', 'black', 'blue', 'green', 'orange', 'purple']
for j in range(K):

    plt.scatter(X[assignments == j, 0], X[assignments == j, 1], color=colors[j], label=f"Cluster {j}")
    
    # Plot centroid j
    plt.scatter(centroids[j, 0], centroids[j, 1], color=colors[j], marker="X", s=200)


plt.title(f"K-Means with K={K}")
plt.legend()
plt.show()







# model = KMeans(n_clusters=2, random_state=42, n_init=10)
# sklearn_labels = model.fit_predict(X)
# plt.scatter(X[sklearn_labels == 0,0], X[sklearn_labels == 0,1],color = "red",label = "cluster 0 ")
# plt.scatter(X[sklearn_labels == 1,0], X[sklearn_labels == 1,1],color = "black",label = "cluster 1 ")
# plt.title(f"K-Means with K={4} using sklearn")
# plt.legend()
# plt.show()
         #for t in range(100):

#     xt= np.sqrt(1-beta[t]) *x_prev+ np.sqrt(beta[t]) * noise
#     xt =np.clip(xt,0,1)
#     all_images.append(xt)

#     cumultive_xt= np.sqrt(at_hat) *x_prev + np.sqrt(1-at_hat) * noise
#     cumaltive_xt = np.clip(cumaltive_xt,0,1)


