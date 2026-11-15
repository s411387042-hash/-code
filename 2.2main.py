import matplotlib.pyplot as plt

def plot_memberships(X, y, theta_values=[0, 0.5, 1.0]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    # 定義三個類別的顏色對應
    colors = {0: 'blue', 1: 'orange', 2: 'green'}
    
    for ax, theta in zip(axes, theta_values):
        model = FuzzyKNN(k=5, theta=theta)
        model.fit(X, y)
        
        for i in range(len(X)):
            true_class = y[i]
            # 信心水準定義為真實類別的隸屬度
            confidence = model.memberships[i][true_class] 
            
            # 使用 alpha 值來對應顏色深淺 (限制在 0.1 到 1.0 之間避免完全透明看不見)
            alpha_val = max(0.1, min(1.0, confidence))
            
            ax.scatter(X[i, 0], X[i, 1], color=colors[true_class], alpha=alpha_val, s=30)
            
        ax.set_title(f'Theta = {theta}')
        ax.set_xlabel('sepal length')
        if theta == 0:
            ax.set_ylabel('sepal width')
            
    plt.tight_layout()
    plt.show()

# 執行繪圖
plot_memberships(X, y)
