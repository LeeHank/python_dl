#!/usr/bin/env python
# coding: utf-8

# ## 自動微分(Automatic Differentiation)

# In[1]:


import torch


# In[2]:


# 設定 x 參與自動微分
x = torch.tensor(4.0, requires_grad=True) 

y = x ** 2            # y = x^2

print(y)
print(y.grad_fn)      # y 梯度函數
y.backward()          # 反向傳導
print(x.grad)         # 取得梯度, dy/dx = 2x, 2x 在 x = 4 的取值 = 8


# ## 取得變數的自動微分相關屬性值

# In[3]:


# 設定變數值
x = torch.tensor(1.0, requires_grad = True)
y = torch.tensor(2.0)
z = x * y
z.retain_grad()

# 顯示自動微分相關屬性
for i, name in zip([x, y, z], "xyz"):
    print(f"{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n" + 
            f"grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n")


# ## 交叉熵(cross entropy)範例

# In[3]:


# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
from IPython.display import Image
Image('./images/cross_entropy.png', width=500)


# * 定義一下上圖的計算圖：
#   * 假設 batch size = 3
#   * 此時 y 的 shape 為 (3,)，例如 [0, 0, 0]，表示有三個樣本，他們的 y 值不是 0 就是 1，剛好這三個樣本都是 0
#   * x 如果代表 5 個 feature，那目前的 shape 為 (5,)
#   * w 

# In[4]:


x = torch.ones(5)  
y = torch.zeros(3)  
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('z 梯度函數：', z.grad_fn)
print('loss 梯度函數：', loss.grad_fn)


# In[5]:


loss.backward()
print(w.grad)         # w梯度值
print(b.grad)         # b梯度值


# In[6]:


# Variable 在 v0.4.0已被棄用，直接使用 tensor 即可
from torch.autograd import Variable
x = Variable(torch.ones(1), requires_grad=True)
y = x + 1
y.backward()
print(x.grad)


# In[3]:


# 替代 Variable
x2 = torch.ones(1, requires_grad=True)
y = x2 + 1
y.backward()
print(x2.grad)


# ## 多次梯度下降

# In[13]:


x = torch.tensor(5.0, requires_grad=True)
y = x ** 3            # y = x^3

y.backward(retain_graph=True) # 梯度下降
print(f'一次梯度下降={x.grad}') 

y.backward(retain_graph=True) # 梯度下降
print(f'二次梯度下降={x.grad}') 

y.backward() # 梯度下降
print(f'三次梯度下降={x.grad}') 


# ## 梯度 reset

# In[12]:


x = torch.tensor(5.0, requires_grad=True)
y = x ** 3            # y = x^3

y.backward(retain_graph=True) # 梯度下降
print(f'一次梯度下降={x.grad}') 
x.grad.zero_()                # 梯度 reset

y.backward(retain_graph=True) # 梯度下降
print(f'二次梯度下降={x.grad}') 
x.grad.zero_()                # 梯度 reset

y.backward() # 梯度下降
print(f'三次梯度下降={x.grad}') 


# ## 多個變數梯度下降

# In[32]:


x = torch.tensor(5.0, requires_grad=True)
y = x ** 3            # y = x^3
z = y ** 2            # z = y^2

z.backward() # 梯度下降
print(f'x 梯度下降 = {x.grad}') # 6 * x^5


# ## 梯度下降法(Gradient Descent)

# In[3]:


# 載入套件
import numpy as np
import matplotlib.pyplot as plt

# 目標函數(損失函數):y=x^2
def func(x): return x ** 2

# 目標函數的一階導數:dy/dx=2*x
def dfunc(x): return 2 * x

# 超參數(Hyperparameters)
x_start = 5     # 起始權重
epochs = 15     # 執行週期數 
lr = 0.3        # 學習率 

# 梯度下降法 
def GD(x_start, df, epochs, lr):    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # 更新 x_new = x — learning_rate * gradient        
        x += - dx * lr         
        xs[i+1] = x    
    return xs

# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr) 
print (np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c='r', marker ='o', markersize=5)    

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 正黑體 
plt.rcParams['axes.unicode_minus'] = False # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()


# ## 使用 PyTorch

# In[15]:


# 載入套件
import torch
import numpy as np
import matplotlib.pyplot as plt

# 目標函數(損失函數):y=x^2
def func(x): return x ** 2

# 自動微分
def dfunc(x): 
    x = torch.tensor(float(x), requires_grad=True)
    y = x ** 2 # 目標函數(損失函數)
    y.backward()
    return x.grad

# 超參數(Hyperparameters)
x_start = 5     # 起始權重
epochs = 15     # 執行週期數 
lr = 0.3        # 學習率 

# 梯度下降法 
def GD(x_start, df, epochs, lr):    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # x更新 x_new = x — learning_rate * gradient        
        x += - dx * lr         
        xs[i+1] = x    
    return xs

# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr) 
print (np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c='r', marker ='o', markersize=5)    

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 正黑體 
plt.rcParams['axes.unicode_minus'] = False # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()


# ## 使用 TensorFlow

# In[4]:


# 載入套件
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 目標函數(損失函數):y=x^2
def func(x): return x ** 2

# 自動微分
def dfunc(x_value): 
    x = tf.Variable(x_value, dtype=tf.float32) # 宣告 TensorFlow 變數(Variable)
    with tf.GradientTape() as g: # 自動微分
        y = x ** 2                # y = x^2
    dy_dx = g.gradient(y, x)     # 取得梯度
    return dy_dx.numpy()         # 轉成 NumPy array

# 超參數(Hyperparameters)
x_start = 5     # 起始權重
epochs = 15     # 執行週期數 
lr = 0.3        # 學習率 

# 梯度下降法 
def GD(x_start, df, epochs, lr):    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # x更新 x_new = x — learning_rate * gradient        
        x += - dx * lr         
        xs[i+1] = x    
    return xs

# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr) 
print (np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c='r', marker ='o', markersize=5)    

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 正黑體 
plt.rcParams['axes.unicode_minus'] = False # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()


# In[17]:


# 損失函數
def func(x): return 2*x**4-3*x**2+2*x-20

# 自動微分
def dfunc(x): 
    x = torch.tensor(float(x), requires_grad=True)
    y = 2*x**4-3*x**2+2*x-20
    y.backward()
    return x.grad

# 超參數(Hyperparameters)
x_start = 5     # 起始權重
epochs = 15000  # 執行週期數 
lr = 0.001      # 學習率 

# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr) 
print (np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c='r', marker ='o', markersize=5)    

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 正黑體 
plt.rcParams['axes.unicode_minus'] = False # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()


# In[ ]:




