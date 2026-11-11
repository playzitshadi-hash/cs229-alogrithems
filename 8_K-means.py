import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

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

plt.scatter(X[:, 0], X[:, 1])
plt.title("Unlabeled dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

new_means=[]
K = 2
random_indices = np.random.choice(len(X), K, replace=False)
centroids = X[random_indices]
for j in range(2): 

  cluster_mean = ((centroids == j)[:, np.newaxis] * X).sum(axis=0) / (centroids == j).sum()
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

model = KMeans(n_clusters=2, random_state=42, n_init=10)
sklearn_labels = model.fit_predict(X)
plt.scatter(X[sklearn_labels == 0,0], X[sklearn_labels == 0,1],color = "red",label = "cluster 0 ")
plt.scatter(X[sklearn_labels == 1,0], X[sklearn_labels == 1,1],color = "black",label = "cluster 1 ")
plt.title(f"K-Means with K={4} using sklearn")
plt.legend()
plt.show()
sklearn_kmeans_ari = adjusted_rand_score(true_labels, sklearn_labels)
print(f"Scikit-Learn K-Means ARI: {sklearn_kmeans_ari}")
print("K-means ARI:",adjusted_rand_score(true_labels,assignments))




