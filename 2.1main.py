import numpy as np
import pandas as pd
import operator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 完整修改後的 FuzzyKNN 類別
# ==========================================
class FuzzyKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3, plot=False, theta=0.49): # [修改] 新增 theta 參數
        self.k = k
        self.plot = plot
        self.theta = theta # [修改] 儲存 theta 參數
        
    def fit(self, X, y=None):
        self._check_params(X,y)
        self.X = X
        self.y = y
        self.xdim = len(self.X[0])
        self.n = len(y)
        
        classes = list(set(y))
        classes.sort()
        self.classes = classes
        
        self.df = pd.DataFrame(self.X)
        self.df['y'] = self.y
        
        self.memberships = self._compute_memberships()
        self.df['membership'] = self.memberships
        self.fitted_ = True
        return self
    
    def _compute_memberships(self):
        memberships = []
        for i in range(self.n):
            x = self.X[i]
            y = self.y[i]
            
            neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
            counts = self._get_counts(neighbors)
            
            membership = dict()
            for c in self.classes:
                # [修改] 改用 dict.get 來安全取得數量，取代原本的 try-except
                count_c = counts.get(c, 0) 
                
                # [修改] 實作公式 (4): (k_j / k) * theta
                uci = (count_c / self.k) * self.theta 
                
                # [修改] 如果是樣本本身的真實類別 (j = i)，加上 -theta + 1
                if c == y:
                    uci += (1 - self.theta) 
                    
                membership[c] = uci
                    
            memberships.append(membership)
        return memberships
        
    def predict(self, X):
        if getattr(self, 'fitted_', None) == None:
            raise Exception('predict() called before fit()')
        else:
            m = 2
            y_pred = []
            
            for x in X:
                neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
                
                votes = {}
                for c in self.classes:
                    den = 0
                    for n in range(self.k):
                        dist = np.linalg.norm(x - neighbors.iloc[n,0:self.xdim])
                        den += 1 / (dist ** (2 / (m-1)))
                    
                    neighbors_votes = []
                    for n in range(self.k):
                        dist = np.linalg.norm(x - neighbors.iloc[n,0:self.xdim])
                        num = (neighbors.iloc[n].membership[c]) / (dist ** (2 / (m-1)))
                        
                        vote = num/den
                        neighbors_votes.append(vote)
                    votes[c] = np.sum(neighbors_votes)
                    
                pred = max(votes.items(), key=operator.itemgetter(1))[0]
                y_pred.append((pred, votes))
                
            return y_pred
        
    def score(self, X, y):
        if getattr(self, 'fitted_', None) == None:
            raise Exception('score() called before fit()')
        else:
            predictions = self.predict(X)
            y_pred = [t[0] for t in predictions]
            return accuracy_score(y_pred=y_pred, y_true=y)
    
    def _find_k_nearest_neighbors(self, df, x):
        X = df.iloc[:,0:self.xdim].values
        df['distances'] = [np.linalg.norm(X[i] - x) for i in range(self.n)]
        df.sort_values(by='distances', ascending=True, inplace=True)
        neighbors = df.iloc[0:self.k]
        return neighbors
                
    def _get_counts(self, neighbors):
        groups = neighbors.groupby('y')
        counts = {group[1]['y'].iloc[0]:group[1].count()[0] for group in groups}
        return counts
        
    def _check_params(self, X, y):
        if type(self.k) != int:
            raise Exception('"k" should have type int')
        elif self.k >= len(y):
            raise Exception('"k" should be less than no of feature sets')
        elif self.k % 2 == 0:
            raise Exception('"k" should be odd')
            
        if type(self.plot) != bool:
            raise Exception('"plot" should have type bool')

# ==========================================
# 2. 驗證腳本 (確認 theta = 0, 0.5, 1 的行為)
# ==========================================
if __name__ == "__main__":
    # 載入 Iris 資料集
    X, y = load_iris(return_X_y=True)
    
    for theta_val in [0, 0.5, 1.0]:
        model = FuzzyKNN(k=5, theta=theta_val)
        model.fit(X, y)
        
        # 檢查所有樣本的隸屬度加總是否為 1 (Membership normalization)
        sums = [sum(m.values()) for m in model.memberships]
        is_normalized = np.allclose(sums, 1.0)
        print(f"Theta={theta_val} | 所有樣本隸屬度總和是否為 1? {is_normalized}")
        
        # 觀察第一個樣本(真實類別為0)的隸屬度分佈，以驗證行為
        print(f"Sample 0 memberships (True Class=0): {model.memberships[0]}")
        print("-" * 50)
