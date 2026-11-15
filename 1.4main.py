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
