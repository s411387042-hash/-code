# ==========================================
# 4. 2.4題 互動式視覺化 (使用 Colab 內建的 ipywidgets)
# ==========================================
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np

# 建立一個繪圖函數，接收 theta_value 作為參數
def interactive_plot(theta_value):
    # 使用你前面已經定義好的 FuzzyKNN 與載入好的 X, y
    model = FuzzyKNN(k=5, theta=theta_value)
    model.fit(X, y)
    
    plt.figure(figsize=(8, 6))
    colors = {0: 'blue', 1: 'orange', 2: 'green'}
    
    for i in range(len(X)):
        true_class = y[i]
        # 信心水準定義為真實類別的隸屬度
        confidence = model.memberships[i][true_class] 
        # 使用 alpha 值來對應顏色深淺
        alpha_val = max(0.1, min(1.0, confidence))
        
        plt.scatter(X[i, 0], X[i, 1], color=colors[true_class], alpha=alpha_val, s=50, edgecolors='none')
        
    plt.title(f'Fuzzy k-NN Membership Confidence (Theta = {theta_value:.1f})')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    
    # 固定 X 軸與 Y 軸的範圍，避免拖動拉桿時畫面跳動
    plt.xlim(4.0, 8.2) 
    plt.ylim(1.8, 4.6)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 呼叫 interact 函數，這會在 Colab 儲存格下方自動生成一個拉桿
interact(interactive_plot, 
         theta_value=widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description='Theta (θ):'))
