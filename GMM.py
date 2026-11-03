import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


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
d = X.shape[1] 
K = 2  

def gaussian_pdf(x, mean, cov, reg_covar=1e-6):
    # Add regularization to prevent singular matrices
    cov_reg = cov + reg_covar * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov_reg)
    inv_cov = np.linalg.inv(cov_reg)
    
    de = 1 / (((2 * np.pi) ** (d / 2)) * np.sqrt(np.exp(logdet)))
    diff = x - mean
    exp = -0.5 * diff.T @ inv_cov @ diff
    return de * np.exp(exp)

# Initialization
np.random.seed(42)
indices = np.random.choice(len(X), 2, replace=False)
mean0 = X[indices[0]].copy()
mean1 = X[indices[1]].copy()
initial_cov = np.cov(X.T) + 1e-6 * np.eye(d)

cov0 = initial_cov.copy()
cov1 = initial_cov.copy()
phi0 = 0.5
phi1 = 0.5

for iteration in range(50):
    responsibilities = np.zeros((len(X), K))  
    log_likelihood = 0
    
    # E-step
    for i in range(len(X)):
        prob0 = gaussian_pdf(X[i], mean0, cov0)
        prob1 = gaussian_pdf(X[i], mean1, cov1)
        
        weighted_prob0 = phi0 * prob0
        weighted_prob1 = phi1 * prob1
        denominator = weighted_prob0 + weighted_prob1 + 1e-10
        
        log_likelihood += np.log(denominator)
        responsibilities[i, 0] = weighted_prob0 / denominator
        responsibilities[i, 1] = weighted_prob1 / denominator

    # M-step with smoothing to prevent zero divisions
    sum_resp0 = np.sum(responsibilities[:, 0]) + 1e-10
    sum_resp1 = np.sum(responsibilities[:, 1]) + 1e-10

    update_phi0 = sum_resp0 / len(X)
    update_phi1 = sum_resp1 / len(X)
    
    update_mean0 = np.sum(responsibilities[:, 0][:, np.newaxis] * X, axis=0) / sum_resp0
    update_mean1 = np.sum(responsibilities[:, 1][:, np.newaxis] * X, axis=0) / sum_resp1
    
    diff0 = X - update_mean0
    diff1 = X - update_mean1
    weighted_diff0 = diff0 * responsibilities[:, 0][:, np.newaxis]
    weighted_diff1 = diff1 * responsibilities[:, 1][:, np.newaxis]
    
    # Update covariances with regularization jitter
    update_cov0 = (weighted_diff0.T @ diff0) / sum_resp0 + 1e-6 * np.eye(d)
    update_cov1 = (weighted_diff1.T @ diff1) / sum_resp1 + 1e-6 * np.eye(d)

    phi0 = update_phi0
    phi1 = update_phi1
    mean0 = update_mean0
    mean1 = update_mean1
    cov0 = update_cov0
    cov1 = update_cov1

    if iteration % 5 == 0:
        print(f"Iteration {iteration}: Log-Likelihood = {log_likelihood:.4f}")

# Final predictions
labels = np.argmax(responsibilities, axis=1)
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], color="red", label="cluster 0")
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], color="black", label="cluster 1")
plt.scatter(mean0[0], mean0[1], color="red", marker="X", s=200, label="Mean 0")
plt.scatter(mean1[0], mean1[1], color="black", marker="X", s=200, label="Mean 1")
plt.legend()
plt.title("Fixed Gaussian Mixture Model")
plt.show()


gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=42
)

gmm_labels = gmm.fit_predict(X)
ari = adjusted_rand_score(gmm_labels, labels)
print("Adjusted Rand Index:", ari)
print(f"Means sk {gmm.means_} gmm {mean0} :: {mean1}")
print("Your weights:", phi0, phi1)
print("Sklearn weights:", gmm.weights_)
print(f"Covariances sk {gmm.covariances_} gmm {cov0}")
print(f"Labels sk {gmm_labels} gmm {labels}")
print(f"Log-Likelihood sk {gmm.score(X) * len(X)} gmm {log_likelihood}")
print("K-means ARI:", adjusted_rand_score(true_labels, assignments))
print("GMM ARI:", adjusted_rand_score(true_labels, labels))
sklearn_kmeans_ari = adjusted_rand_score(true_labels, sklearn_labels)
print(f"Scikit-Learn K-Means ARI: {sklearn_kmeans_ari}")
sklearn_gmm_ari = adjusted_rand_score(true_labels, gmm_labels)
print(f"Scikit-Learn GMM ARI: {sklearn_gmm_ari}")