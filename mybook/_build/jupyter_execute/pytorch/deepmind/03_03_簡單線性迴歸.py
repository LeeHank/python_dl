#!/usr/bin/env python
# coding: utf-8

# # 使用自動微分估算簡單線性迴歸的參數(w、b)
# ## $ y = wx + b$

# In[4]:


# 載入套件
import numpy as np 
import torch


# ## 定義訓練函數

# In[74]:


def train(X, y, epochs=100, lr=0.0001):
    loss_list, w_list, b_list=[], [], []
    
    # w、b 初始值均設為常態分配之隨機亂數 
    w = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)    
    for epoch in range(epochs):   # 執行訓練週期       
        y_pred = w * X + b        # 預測值
        
        # 計算損失函數值
        MSE = torch.square(y - y_pred).mean()        
        MSE.backward()      
        
        # 設定不參與梯度下降，w、b才能運算
        with torch.no_grad():
            # 新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
            w -= lr * w.grad
            b -= lr * b.grad 
        
        # 記錄訓練結果
        if (epoch+1) % 1000 == 0 or epochs < 1000:
            # detach：與運算圖分離，numpy()：轉成陣列
            # w.detach().numpy()
            w_list.append(w.item())  # w.item()：轉成常數
            b_list.append(b.item())
            loss_list.append(MSE.item())
        
        # 梯度重置
        w.grad.zero_()
        b.grad.zero_()
        
    return w_list, b_list, loss_list


# ## 產生隨機資料

# In[75]:


# 產生線性隨機資料100筆，介於 0-50
n = 100
X = np.linspace(0, 50, n) 
y = np.linspace(0, 50, n) 
  
# 資料加一點雜訊(noise)
X += np.random.uniform(-10, 10, n) 
y += np.random.uniform(-10, 10, n) 


# ## 執行訓練

# In[86]:


# 執行訓練
w_list, b_list, loss_list = train(torch.tensor(X), torch.tensor(y), epochs=100000)

# 取得 w、b 的最佳解
print(f'w={w_list[-1]}, b={b_list[-1]}')


# In[93]:


# 執行訓練
w_list, b_list, loss_list = train(torch.tensor(X), torch.tensor(y))

# 取得 w、b 的最佳解
print(f'w={w_list[-1]}, b={b_list[-1]}')


# ## 以NumPy驗證

# In[94]:


# 執行訓練
coef = np.polyfit(X, y, deg=1)

# 取得 w、b 的最佳解
print(f'w={coef[0]}, b={coef[1]}')


# In[95]:


from sklearn.linear_model import LinearRegression

X2 = X.reshape(X.shape[0], 1)

lr = LinearRegression()
lr.fit(X2, y)

lr.coef_[0], lr.intercept_


# ## 顯示迴歸線

# In[96]:


import matplotlib.pyplot as plt 

plt.scatter(X, y, label='data')
plt.plot(X, w_list[-1] * X + b_list[-1], 'r-', label='predicted')
plt.legend()


# In[97]:


# NumPy 求得的迴歸線
import matplotlib.pyplot as plt 

plt.scatter(X, y, label='data')
plt.plot(X, coef[0] * X + coef[1], 'r-', label='predicted')
plt.legend()


# In[81]:


# 損失函數繪圖
plt.plot(loss_list)


# In[98]:


loss_list


# In[99]:


w_list


# In[ ]:




