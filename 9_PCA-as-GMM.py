import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
plt.plot(range(1, 11), explained_variance_ratio, marker='o')
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained Ratio")
plt.show()


print("First 5 projections:", a[:5])
plt.hist(a, bins=30)
plt.title("Histogram of Projections on PC1")
plt.show()


reconstruction = np.outer(a, u)

MSE = np.mean(np.sum((centered_X - reconstruction) ** 2, axis=1))
print("Reconstruction MSE:", MSE)
U2 = eigenvectors[:, sort[:2]]

projected_2d = centered_X @ U2
reconstruction_2d = projected_2d @ U2.T

mse_2 = np.mean(np.sum((centered_X - reconstruction_2d) ** 2,axis=1))

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Reconstruction MSE using sklearn PCA: ", np.mean((X - X_reconstructed) ** 2))
print("Reconstruction MSE From scratch PCA: ", MSE)
print("From scratch PCA, 2 components:", mse_2)
print("Variance retained by 2 components:", np.sum(explained_variance_ratio[:2]) * 100, "%")






