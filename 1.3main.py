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
