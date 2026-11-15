import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import eigh

# --- 1.1 Data Generation ---
np.random.seed(42)

# Generate orthogonal unit vectors u and v in R^6
random_matrix = np.random.randn(6, 2)
q, r = np.linalg.qr(random_matrix)
u, v = q[:, 0], q[:, 1]

# Verify orthogonality and unit-length
print(f"||u|| = {np.linalg.norm(u):.4f} (Expected: 1.0)")
print(f"||v|| = {np.linalg.norm(v):.4f} (Expected: 1.0)")
print(f"u^T v = {np.dot(u, v):.4e} (Expected: ~0.0)")

sigma_b = 0.5
sigma_e = 0.2
mu_classes = [-2, 0, 2]
sigma_a_list = [1, 3, 10]
n_samples_per_class = 150

def generate_dataset(sigma_a):
    X, Y = [], []
    for c_idx, mu_c in enumerate(mu_classes):
        # a ~ N(0, sigma_a^2)
        a = np.random.normal(0, sigma_a, n_samples_per_class)
        # b_c ~ N(mu_c, sigma_b^2)
        b = np.random.normal(mu_c, sigma_b, n_samples_per_class)
        # epsilon ~ N(0, sigma_e^2 I)
        epsilon = np.random.normal(0, sigma_e, (n_samples_per_class, 6))
        
        # x = au + b_c v + epsilon
        x = np.outer(a, u) + np.outer(b, v) + epsilon
        X.append(x)
        Y.append(np.full(n_samples_per_class, c_idx))
    return np.vstack(X), np.hstack(Y)

datasets = {}
for sa in sigma_a_list:
    datasets[sa] = {
        'train': generate_dataset(sa),
        'test': generate_dataset(sa)
    }

# Illustrate 3-class training sets with randomly selected 3 dimensions (e.g., dims 0, 1, 2)
fig = plt.figure(figsize=(15, 5))
for i, sa in enumerate(sigma_a_list):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    X_tr, Y_tr = datasets[sa]['train']
    scatter = ax.scatter(X_tr[:, 0], X_tr[:, 1], X_tr[:, 2], c=Y_tr, cmap='viridis', alpha=0.6)
    ax.set_title(f"Train Set (sigma_a={sa})\nDims: 0, 1, 2")
plt.tight_layout()
# 改成存檔：
fig.savefig('1_1_3D_scatter.png')
plt.close(fig) # 存檔完關閉畫布釋放記憶體
