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

# --- 1.2 PCA Implementation ---
def apply_pca(X_train, X_test, n_components):
    mean = np.mean(X_train, axis=0)
    X_train_centered = X_train - mean
    X_test_centered = X_test - mean
    
    cov_matrix = np.cov(X_train_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    
    projection_matrix = sorted_eigenvectors[:, :n_components]
    return X_train_centered.dot(projection_matrix), X_test_centered.dot(projection_matrix)

# --- 1.3 LDA Implementation ---
def apply_lda(X_train, y_train, X_test, n_components):
    n_features = X_train.shape[1]
    class_labels = np.unique(y_train)
    
    mean_overall = np.mean(X_train, axis=0)
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))
    
    for c in class_labels:
        X_c = X_train[y_train == c]
        mean_c = np.mean(X_c, axis=0)
        # Within-class scatter
        Sw += (X_c - mean_c).T.dot(X_c - mean_c)
        # Between-class scatter
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
        Sb += n_c * (mean_diff).dot(mean_diff.T)
        
    # Generalized eigenvalue problem: Sb*w = lambda*Sw*w
    eigenvalues, eigenvectors = eigh(Sb, Sw)
    
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    
    projection_matrix = sorted_eigenvectors[:, :n_components]
    return X_train.dot(projection_matrix), X_test.dot(projection_matrix)

# --- 1.4 & 1.5 & 1.6 Projection, k-NN and Visualization ---
results = {
    'PCA-1D': [], 'PCA-2D': [],
    'LDA-1D': [], 'LDA-2D': []
}

fig_pca, axes_pca = plt.subplots(1, 3, figsize=(15, 4))
fig_lda, axes_lda = plt.subplots(1, 3, figsize=(15, 4))
fig_pca.suptitle('PCA-2D Projection')
fig_lda.suptitle('LDA-2D Projection')

for i, sa in enumerate(sigma_a_list):
    X_tr, Y_tr = datasets[sa]['train']
    X_te, Y_te = datasets[sa]['test']
    
    # 1.4 Projections
    X_tr_pca1, X_te_pca1 = apply_pca(X_tr, X_te, 1)
    X_tr_pca2, X_te_pca2 = apply_pca(X_tr, X_te, 2)
    X_tr_lda1, X_te_lda1 = apply_lda(X_tr, Y_tr, X_te, 1)
    X_tr_lda2, X_te_lda2 = apply_lda(X_tr, Y_tr, X_te, 2)
    
    # Scatter plots for 2D
    axes_pca[i].scatter(X_te_pca2[:, 0], X_te_pca2[:, 1], c=Y_te, cmap='viridis', alpha=0.7)
    axes_pca[i].set_title(f'Test Set: sigma_a={sa}')
    
    axes_lda[i].scatter(X_te_lda2[:, 0], X_te_lda2[:, 1], c=Y_te, cmap='viridis', alpha=0.7)
    axes_lda[i].set_title(f'Test Set: sigma_a={sa}')
    
    # 1.5 k-NN classification
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_tr_pca1, Y_tr)
    results['PCA-1D'].append(knn.score(X_te_pca1, Y_te))
    
    knn.fit(X_tr_pca2, Y_tr)
    results['PCA-2D'].append(knn.score(X_te_pca2, Y_te))
    
    knn.fit(X_tr_lda1, Y_tr)
    results['LDA-1D'].append(knn.score(X_te_lda1, Y_te))
    
    knn.fit(X_tr_lda2, Y_tr)
    results['LDA-2D'].append(knn.score(X_te_lda2, Y_te))

# 把迴圈後的 plt.show() 改成下面這樣：
fig_pca.tight_layout()
fig_lda.tight_layout()
fig_pca.savefig('1_4_PCA_2D_Projection.png')
fig_lda.savefig('1_4_LDA_2D_Projection.png')
plt.close(fig_pca)
plt.close(fig_lda)

# 1.6 Results output and Accuracy plot
print("\n--- 5x4 Accuracy Table ---")
print(f"{'Method':<10} | {'sa=1':<8} | {'sa=3':<8} | {'sa=10':<8}")
print("-" * 40)
for method, accs in results.items():
    print(f"{method:<10} | {accs[0]:.4f} | {accs[1]:.4f} | {accs[2]:.4f}")

plt.figure(figsize=(8, 5))
for method, accs in results.items():
    plt.plot(sigma_a_list, accs, marker='o', label=method)
plt.title('Test Accuracy vs sigma_a')
plt.xlabel('sigma_a')
plt.ylabel('Accuracy')
plt.xticks(sigma_a_list)
plt.legend()
plt.grid(True)

# 改成存檔：
plt.savefig('1_6_Accuracy_Plot.png')
plt.close()
