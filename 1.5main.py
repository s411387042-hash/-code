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
