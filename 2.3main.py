# ==========================================
# 3. 2.3題 參數敏感度與解釋 (Parameter Sensitivity)
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    
    # 挑選三個代表性樣本 (依照 Iris 資料集的特性)
    # Index 0: Setosa (核心區，無重疊)
    # Index 70: Versicolor (位於與 Virginica 的交界區)
    # Index 119: Virginica (位於與 Versicolor 的交界區，有時會被誤判)
    sample_indices = [0, 70, 119]
    theta_values = [0.0, 0.5, 1.0]
    
    print(f"{'Sample (True)':<15} | {'Theta':<5} | {'Class 0':<7} | {'Class 1':<7} | {'Class 2':<7} | {'Prediction'}")
    print("-" * 65)
    
    for idx in sample_indices:
        x_sample = X[idx]
        true_label = y[idx]
        
        for theta in theta_values:
            model = FuzzyKNN(k=5, theta=theta)
            model.fit(X, y)
            
            # 取得該樣本在訓練後的隸屬度
            memberships = model.memberships[idx]
            
            # 進行預測 (predict 預期輸入 2D array，因此加 []，並取第一個預測結果)
            prediction_output = model.predict([x_sample])[0]
            predicted_label = prediction_output[0]
            
            print(f"Index {idx} ({true_label})   | {theta:<5} | {memberships.get(0, 0):.4f}  | {memberships.get(1, 0):.4f}  | {memberships.get(2, 0):.4f}  | {predicted_label}")
        print("-" * 65)
