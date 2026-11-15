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
