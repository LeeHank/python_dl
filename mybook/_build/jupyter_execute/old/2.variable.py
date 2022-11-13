#!/usr/bin/env python
# coding: utf-8

# # Introduction to Variables

# * 上一節介紹的 tensor，其實就是 `tf.constant()` 的結果，特點是 immutable，所以你無法去改值 
# * 例如： `my_tensor = tf.constant([1,2,3])`, 然後 `my_tensor[0] = 1` 這是不行的
# * 但在 ML 中，有些 tensor，我會希望可以改值，例如 model 的 weight，在每次學習過程中，都需要更新。那 weight 就無法用 tensor 來處理。
# * 這時候， `tf.Variable` 這個 class，就可以把它想成 mutable 的 tensor， A `tf.Variable` represents a tensor whose value can be changed by running ops on it.  所以可以幫助我們 shared, persistent state your program manipulates. 
# * `tf.keras` 就是用 `tf.Variable` 來儲存 model parameters. 
# * 而這一章，就是要來講，如何 create, update, 以及 manage instances of `tf.Variable` in TensorFlow.

# ## Setup

# In[1]:


import tensorflow as tf

# Uncomment to see where your variables get placed (see below)
# tf.debugging.set_log_device_placement(True)


# ## Create a variable

# In[5]:


# tf.constant
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# tf.variable
my_variable = tf.Variable(my_tensor)

my_variable


# * 可以看到，是 `tf.Variable` 這個 class  
# * 而 `tf.Variable` 其實就是 mutable 的 tensor，所以他也有 `dtype` 和 `shape` 兩個 attribute

# In[6]:


print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)


# ## assign

# * 來看一下他和 tensor 最大的不同

# In[11]:


# tensor (i.e. tf.constant) 是 immutable
my_tensor[0,0] = 99.99


# In[15]:


# variable 是 mutable，用 `.assign()` 來加入新值
my_variable[0,0].assign(99.99) # 當下就更新了，不需要 my_variable = my_variable[0,0].assign(99.99)


# * 那在模型訓練時，更常用的，是把整個 tf.Variable 裡面的所有值都替換 (e.g. 權重更新)

# In[20]:


a = tf.Variable([2.0, 3.0])

# This will keep the same dtype, float32
a.assign([1, 2]) 


# * 其他用法，例如 python，在更新資料時，會用
#   * `a = a + 1` or `a += 1`. 
#   * `b = b - 1` or `b -= 1`. 
# * 那在 variable 中，要用 `assign_add` 和 `assign_sub`，例如：
#   * `a.assign_add(1)`. 
#   * `b.assign_sub(1)`. 

# In[22]:


a = tf.Variable([2.0, 3.0])

print(a.assign_add([5,6]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]


# * 最後，要注意的是，一但宣告了 Variable，他就幫你在記憶體中開一個位子了，所以，你不能塞不同 size 的東西進去：

# In[21]:


# Not allowed as it resizes the variable: 
try:
  a.assign([1.0, 2.0, 3.0])
except Exception as e:
  print(f"{type(e).__name__}: {e}")


# ## convert

# * variable 可以用. 
#   * `tf.convert_to_tensor()` 轉成 tensor (i.e. tf.constant)
#   * 用 `.numpy()` 轉成 numpy  

# In[23]:


# 轉成 tensor
print(my_variable)
print(tf.convert_to_tensor(my_variable))


# In[19]:


# 轉成 numpy
print(my_variable)
print(my_variable.numpy())


# ## Lifecycles, naming, and watching

# * In Python-based TensorFlow, `tf.Variable` instance have the same lifecycle as other Python objects. 
# * When there are no references to a variable it is automatically deallocated.
# * Variables 可以被命名，這樣可以幫助我們 track and debug  
# * 而且，不同 variables 可以命相同的名字(只是可以而已，不要真的這麼做，會混淆自己)  

# In[24]:


# Create a and b; they will have the same name but will be backed by
# different tensors.
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")

# These are elementwise-unequal, despite having the same name
print(a == b)


# * 至於，為啥 variable 要命名？  這是因為在做 training 時， model 的每個 layer ，都會有自己的 name，這樣要 reload 之前存好的 weight 時，他會用 name 去對應和 recover
# * 通常，你也不用真的去命名，因為他會自動幫你命名一個 unique name  。
# * 另外，有些 variable 需要被微分，但有些不用 (例如 counter)，我們可以將 variable 中的 `trainable` 參數定為 False，之後就不會被自動求導。例如底下的例子：

# In[25]:


step_counter = tf.Variable(1, trainable=False)


# ## Placing variables and tensors

# * 通常來說，為了較好的計算速度， tensorflow 會將 tensors 和 variables 放到最快的裝置上做計算，也就是說，大多數的 variables 都會被放到 GPU 上。
# * 但你也可以 override 這個設定，下面給的例子，就是強迫把底下的計算，都放在 CPU 上執行

# In[26]:


with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)


# * 注意，儘管剛剛在設定是 ok 的，但還是建議使用 [distribution strategies](distributed_training.ipynb) 的建議，讓你的計算能被最佳化。

# * 此外，你也可以將 tensor or variable "放在" 一個 device (e.g. CPU)，計算的時後再放到另一個 device (e.g. GPU)
# * 這樣做會造成一些 delay，因為 data needs to be copied between the devices.
# * 然而，你還是有機會要做這件事，例如，you had multiple GPU workers but only want one copy of the variables.
# * 底下來個例子：

# In[11]:


with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)


# Note: Because `tf.config.set_soft_device_placement` is turned on by default, even if you run this code on a device without a GPU, it will still run.  The multiplication step will happen on the CPU.
# 
# For more on distributed training, refer to the [guide](distributed_training.ipynb).

# ## Next steps
# 
# To understand how variables are typically used, see our guide on [automatic differentiation](autodiff.ipynb).
