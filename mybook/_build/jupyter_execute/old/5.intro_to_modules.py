#!/usr/bin/env python
# coding: utf-8

# # Introduction to modules, layers, and models

# * 在用 tensorflow 做 machine learning的時候，會需要去 define, save, and restore a model.
# * 在 tf 中， model 可被定義為： 
#   * A function that computes something on tensors (a **forward pass**)  
#   * Some variables that can be updated in response to training  
# * 在這份文件中，將 go below the surface of Keras to see how TensorFlow models are defined. 
# * This looks at how TensorFlow collects variables and models, as well as how they are saved and restored.

# ## Setup

# In[1]:


import tensorflow as tf
from datetime import datetime

get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## `tf.Module`: Defining models and layers in TensorFlow

# * 大部分的 model 都是由 layers 所組成. 
# * Layers 其實就是 functions，而這個 function 是由可被重複使用的數學結構所定義，裡面有可訓練的變數。  
# * 在 tensorflow 中，大部分 high-level 的 layers 和 models，都是built on the same foundational class: `tf.Module`.

# ### toy example

# In[2]:


class SimpleModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    self.a_variable = tf.Variable(5.0, name="train_me")
    self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
  def __call__(self, x):
    return self.a_variable * x + self.non_trainable_variable

simple_module = SimpleModule(name="simple")

simple_module(tf.constant(5.0))


# * 來看一下，我們剛剛定義了一個 module (你可以叫他 module，也可以叫他 layer，都可以)，他最一開始就繼承了 `tf.Moudle` 這個 class
# * 可以看到，這個 class，有 `__call__` 這個 method，這就是一般 python callable 的定義方式，沒什麼特別
# * 再來看一下 `__init__` 裡面，定義了兩個會用的屬性，也就是兩個 `tf.Variable`。其中一個是 trainable (可微分)，另一個是不給 train (不可微分)
# * 然後，這個 module 要做的事情，就是，當你輸入 x 時，他會幫你乘上一個數，然後再加上一個數
# * 從結果來看，輸入 5 後，得到 32
# * 那，繼承 `tf.Module` 這個 class 有什麼好處？好處就是，已經有寫好一些 method 和 attribute，會幫你省很多力氣。例如，他會自動幫你蒐集總共定義了多少個 tf.Variables，以及， trainable 的 variable 有哪些，

# In[4]:


# all variables
simple_module.variables


# * 可以看到，他用一個 tuple，把所有 variable 給蒐集起來。名稱分別是 `train_me` 和 `do_not_train_me`. 
# * 另外，自動微分最常用的，就是抓出 trainable variables 來做微分。這他也幫你蒐集好了：

# In[5]:


# trainable variable
simple_module.trainable_variables


# ### 自訂 layer

# * 接著，我們可以來定義一個自己的 `Dense` (linear) layer: 

# In[6]:


class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    # Dense layer 的第一個參數，是 weight 矩陣 `w`，起始值用 normal 來生，shape 是 in_features x out_features
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    # Dense layer 的第二個參數，是 bias 向量 `b`，起始值給 0 向量， shape 為 1 x out_features
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    # 當我們 call Dense 的時候，就是輸入一個 x tensor (shape 為 1 x in_features)，然後去計算 x w + b，得到 shape 為 1 x out_features 的向量 
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


# * 可以想像， w 和 b 都是 trainable variable，之後就要靠自動微分來更新 w 和 b 的值

# In[8]:


dense_1 = Dense(in_features = 3, out_features = 6)
dense_1.trainable_variables


# * 可以看到，初始化後的 dense_1，有兩個 trainable variable，而且起始 weight 也列出來給你看了 
# * 可以用用看這個 dense 層的功能：輸入 1 x in_features 的 tensor，輸出 1 x out_features 的 tensor

# In[15]:


x = tf.constant([[1.0, 2.1, 3.2]])
out = dense_1(x)
out


# * 可以看到， output 是一個 1x6 的 tensor

# ### 自訂 model

# * 那如果我們想自定一個 model (從 input 一個 tensor，到 output 一個結果出來。重點在定義中間的 forward propagation 過程)，就可以沿用剛剛的定義方式。
# * 假設，我想做一個 2 個 dense 的 NN (都是linear)，那我可以這樣定義：

# In[16]:


class SequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    # 在這邊定義好，我等等會用到的 layer 有哪些
    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  def __call__(self, x):
    # forward logic
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))


# `tf.Module` instances will automatically collect, recursively, any `tf.Variable` or `tf.Module` instances assigned to it. This allows you to manage collections of `tf.Module`s with a single model instance, and save and load whole models.

# In[7]:


print("Submodules:", my_model.submodules)


# In[8]:


for var in my_model.variables:
  print(var, "\n")


# ### Waiting to create variables

# * 剛剛可以看到，我們定義的 dense ，要定義 input tensor 的 shape (i.e. in_features 是多少
# * 這有點麻煩，而且和 keras 內建的 `tf.keras.layers.Dense` 不同
# * keras 的 Dense 只要知道 out_features 就好， in_features 他會直接去讀你丟給他的 input tensor 來決定
# * 所以，我們稍微改一下原本的 code，就可以做到這件事：

# In[17]:


class FlexibleDenseModule(tf.Module):
  # Note: No need for `in_features`
  def __init__(self, out_features, name=None):
    super().__init__(name=name)
    self.out_features = out_features
    # 加入這行
    self.is_built = False # 起始狀態時，是 False
    
  def __call__(self, x):
    # 第一次 call 時 (self.is_built 為 False 時)
    if not self.is_built:
      # 在這裡才定義 w 和 b
      self.w = tf.Variable(
        tf.random.normal([x.shape[-1], self.out_features]), name='w')
      self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
      self.is_built = True # 並修改狀態

    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


# In[18]:


# Used in a module
class MySequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = FlexibleDenseModule(out_features=3)
    self.dense_2 = FlexibleDenseModule(out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

my_model = MySequentialModule(name="the_model")
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))


# ### Save & load

# * 剛剛不管是自己定義的 layer，或是 model，因為都繼承自 `tf.Module`，所以都可以追蹤到所有的 trainable variable 的 weights
# * 那在訓練過程中，我們當然就可以把這些 weights 給存下來，之後就可以從這個 weight 繼續 train 下去  
# * 存檔的方式有兩種：
#   * checkpoint: 這就是只有存 weight，沒有存 module/layer structure。所以之後讀取時，要先建立一個一樣 module/layer structure 的 object 後，再把存好的 weight 塞回去
#   * SaveModel: 這是把 module/layer structure 以及 對應的 weight，全都存下來。之後只要 load 這個 model 就好，大師兄就全都回來了。

# #### checkpoints (save & load weights)

# * 作法如下：
#   * 先用 `checkpoint = tf.train.Checkpoint(model = my_model_obj)`，來建立一個 checkpoint 物件
#   * 再用 `checkpoint.write('checkpoint_path')`，來把 checkpoint 檔存出來

# In[20]:


# 先建立一個 checkpoint 物件，他追蹤的是我剛剛 train 到一半的 my_model
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint


# In[21]:


# 再把結果寫出來
chkp_path = "my_checkpoint"
checkpoint.write(chkp_path)


# * 看一下資料夾，寫出兩個檔案：  
#   * data 本身 (i.e. my_checkpoint.data-00000-of-00001)。裡面包含 variable values and their attribute lookup paths
#   * index file for metadata (i.e. my_checkpoint.index)。功能是 keeps track of what is actually saved and the numbering of checkpoints

# In[22]:


get_ipython().system('ls my_checkpoint*')


# You can look inside a checkpoint to be sure the whole collection of variables is saved, sorted by the Python object that contains them.

# In[13]:


tf.train.list_variables(chkp_path)


# During distributed (multi-machine) training they can be sharded,  which is why they are numbered (e.g., '00000-of-00001').  In this case, though, there is only have one shard.

# * 如果之後要把 weight 給 load 進來，那做法是：
#   * 先建立一個和之前一樣架構的 model/layer. 
#   * 建立 checkpoint 物件，去追蹤這個新建立好的 model/layer
#   * 用這個新的 checkpoint 物件的 `restore('')` method，把剛剛的 weight 給 load 回來：

# In[23]:


new_model = MySequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore("my_checkpoint")

# Should be the same result as above
new_model(tf.constant([[2.0, 2.0, 2.0]]))


# #### `SavedModel`

# * 另一種儲存的方式，是直接把 model 的 structure 和 weight 全都存起來  
# * 作法是：
#   * 用 `tf.saved_model.save(my_model_obj, "a_folder_path_to_save_model")`. 
#   * 用 `new_model = tf.saved_model.load("a_folder_path_to_save_model")` 來讀檔
# * 看一下範例：

# In[24]:


tf.saved_model.save(my_model, "the_saved_model")


# * 可以看到，檔案被存到 `the_saved_model` 這個資料夾中。
# * 看一下這個資料夾裡有什麼東西：  
#   * `assets` 資料夾： 空的
#   * `variables` 資料夾
#     * variables.data-00000-of-00001: 這就是剛剛 checkpoint 的 data 檔
#     * variables.index: 這就是剛剛 checkpoint 的 index 檔
#   * `saved_model.pb` 檔案： a [protocol buffer](https://developers.google.com/protocol-buffers) describing the functional `tf.Graph`.

# In[19]:


# Inspect the SavedModel in the directory
get_ipython().system('ls -l the_saved_model')


# * 讀檔時，這樣讀：

# In[25]:


new_model = tf.saved_model.load("the_saved_model")


# `new_model`, created from loading a saved model, is an internal TensorFlow user object without any of the class knowledge. It is not of type `SequentialModule`.

# In[26]:


isinstance(new_model, SequentialModule)


# This new model works on the already-defined input signatures. You can't add more signatures to a model restored like this.

# In[27]:


print(my_model([[2.0, 2.0, 2.0]]))
print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))


# Thus, using `SavedModel`, you are able to save TensorFlow weights and graphs using `tf.Module`, and then load them again.

# ## Keras models and layers

# * 剛剛教了用 `tf.Module` 來建立 layer 和 model  
# * 這邊開始，要來看 Keras 是怎麼用 `tf.Module` 的

# ### Keras layers

# * `tf.keras.layers.Layer` 是 Keras 的 layer 的 base class，他繼承自 `tf.Module`. 
# * 所以，我們剛剛是在 `tf.Module` 的基礎下，建立自己的 layer。現在，可以在 `tf.keras.layers.Layer` 的基礎下，建立 layer。  
# * 那這樣的好處是，就可以繼承更多 keras 的 layer 所擁有的 attribute, methods，使得，未來用 keras 的 fit, compile 等功能時，他去吃你定義的 dense，都能取得他預期要拿到的東西  
# * 換句話說，如果你後續想用 keras 的 compile, fit 等功能，那你的 layer，必須繼承 keras layer 的 class，而不是 `tf.Module`. 
# * 我們來看已下範例，寫法基本上和剛剛沒差別，只有 `__call__` 要改成 `call`，因為 keras 自己有定義 call method

# In[28]:


class MyDense(tf.keras.layers.Layer):
  # Adding **kwargs to support base Keras layer arguments
  def __init__(self, in_features, out_features, **kwargs):
    super().__init__(**kwargs)

    # This will soon move to the build step; see below
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def call(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

simple_layer = MyDense(name="simple", in_features=3, out_features=3)


# Keras layers have their own `__call__` that does some bookkeeping described in the next section and then calls `call()`. You should notice no change in functionality.

# In[29]:


simple_layer([[2.0, 2.0, 2.0]])


# ### The `build` step

# * 再來，是 keras 的 layer 寫法中，很重要的 `build` step. 
# * 還記得前面用 `tf.Module` 來寫 layer 時，為了不要每次都定義 input_feature，所以會先寫一個 `self.is_built = False` 來說明，目前還沒被 build。然後當 input tensor 進來後，才定義 w 和 b 的 shape，並把 `self.is_built` 改為 true，表示已經 build 完
# * 那，keras 這邊，就直接定義一個 method 叫 build，就是直接用來取 input_feature 用的
# * `build` is called exactly once, and it is called with the shape of the input. It's usually used to create variables (weights).
# * 我們來改寫一下剛剛的 layer

# In[31]:


class FlexibleDense(tf.keras.layers.Layer):
  # Note the added `**kwargs`, as Keras supports many arguments
  def __init__(self, out_features, **kwargs):
    super().__init__(**kwargs)
    self.out_features = out_features

  def build(self, input_shape):  # Create the state of the layer (weights)
    self.w = tf.Variable(
      tf.random.normal([input_shape[-1], self.out_features]), name='w')
    self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.matmul(inputs, self.w) + self.b


# * 上面的寫法，有幾個重點要注意：  
#   * `__init__` 裡面，多了 `**kwargs**`，因為，繼承自 `tf.keras.layers.Layer` 時，他已經 support 很多其他的 input arguments 了. 
#   * `build(self, input_shape)` 這個 method中，input_shape 是 keras 會自動幫你讀出 input tensor 的 shape。所以下面就會用 input_shape[-1] 來當作 input_feature 數。
#   * `build` 這個 method，你不會真的拿來用，他就是一個輔助 method，當你第一次把 input tensor 丟進去時，他會自己啟用
# * 接著，我們實例化這個 class

# In[32]:


# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)


# * 目前為止，我們還沒丟 input tensor 進去，所以 build 方法還沒被啟用，這時候，我們的 layer，還沒追蹤到任何 variable:

# In[33]:


flexible_dense.variables


# * 一旦我丟一個 input tensor 進去後，就會啟用 build 方法，weight 就被 initialize 了：

# In[34]:


# Call it, with predictably random results
print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))


# In[35]:


flexible_dense.variables


# * 由於 `build` method 只會被 call 一次，所以如果之後 input 的 tensor，shape 和 起始話的時候不同，那就會報 error

# In[36]:


try:
  print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
except tf.errors.InvalidArgumentError as e:
  print("Failed:", e)


# ### Keras models

# * 前面在定義 layer 和 model 時，都是繼承自 `tf.Module` 這個 class  
# * 但當我們要定義 custom keras layer 時，我們會繼承自 `tf.keras.layers.Layer`(此 class 繼承自 `tf.Module`)，這樣就能保有很多 keras layer 的好的特性
# * 同樣的，當我們要定義 custom keras model 時，我們也會繼承一個 keras 的 class，就是 `tf.keras.Model` (此 class 繼承自 `tf.keras.layers.Layer`)  
# * 這樣做的好處是，我們定義好的 Model，可以輕易的被 used, nested, and saved in the same way as Keras layers.
# * 而且，還會有許多 extra functionality that makes them easy to train, evaluate, load, save, and even train on multiple machines.
# * 來寫吧：

# In[37]:


class MySequentialModel(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.dense_1 = FlexibleDense(out_features=3)
    self.dense_2 = FlexibleDense(out_features=2)
  def call(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)


# * 可以看到，比剛剛寫 layer 簡單，因為不用寫 `build` method。這部分在定義 custom layer 時已經做完了。只要單純的 `__init__` 和 `call` 就好
# * 實例化這個 model，一樣的，一開始找不到 weight，因位還沒有 tensor 被餵進來，還不知道 input shape，就不會 initialze weights

# In[38]:


# You have made a Keras model!
my_sequential_model = MySequentialModel(name="the_model")
my_sequential_model.variables


# * 丟個 input tensor 進去，就可以看到結果，以及 variables 了

# In[39]:


# Call it on a tensor, with random results
print("Model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))


# In[40]:


my_sequential_model.variables


# In[41]:


my_sequential_model.submodules


# In[42]:


my_sequential_model.summary()


# ### functional API

# * 另外一種建立 model 的方式，是直接用 functional API，這不僅可以幫我們減少一些時間，也可以獲得一些額外的好處 (e.g. model.summary()時，可發現 output shape 都跑出來了)  
# * functional API 和剛剛 subclass 的寫法，最大差別在，你要先定義 input 的 shape (by `tf.keras.Input(shape = [3,])`)
# * The `input_shape` argument in this case does not have to be completely specified; you can leave some dimensions as `None`.
# * Note: You do not need to specify `input_shape` or an `InputLayer` in a subclassed model; these arguments and layers will be ignored.

# In[44]:


inputs = tf.keras.Input(shape=[3,])
x = FlexibleDense(3)(inputs)
x = FlexibleDense(2)(x)

my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)

my_functional_model.summary()


# In[45]:


my_functional_model(tf.constant([[2.0, 2.0, 2.0]]))


# ## Saving Keras models
# 
# Keras models can be checkpointed, and that will look the same as `tf.Module`.
# 
# Keras models can also be saved with `tf.saved_model.save()`, as they are modules.  However, Keras models have convenience methods and other functionality:

# In[46]:


my_sequential_model.save("exname_of_file")


# Just as easily, they can be loaded back in:

# In[47]:


reconstructed_model = tf.keras.models.load_model("exname_of_file")


# Keras `SavedModels` also save metric, loss, and optimizer states.
# 
# This reconstructed model can be used and will produce the same result when called on the same data:

# In[48]:


reconstructed_model(tf.constant([[2.0, 2.0, 2.0]]))


# There is more to know about saving and serialization of Keras models, including providing configuration methods for custom layers for feature support. Check out the [guide to saving and serialization](https://www.tensorflow.org/guide/keras/save_and_serialize).

# # What's next
# 
# If you want to know more details about Keras, you can follow the existing Keras guides [here](./keras/).
# 
# Another example of a high-level API built on `tf.module` is Sonnet from DeepMind, which is covered on [their site](https://github.com/deepmind/sonnet).

# ### 待完成內容：

# Keras layers have a lot more extra features including:
# 
# * Optional losses
# * Support for metrics
# * Built-in support for an optional `training` argument to differentiate between training and inference use
# * `get_config` and `from_config` methods that allow you to accurately store configurations to allow model cloning in Python
# 
# Read about them in the [full guide](./keras/custom_layers_and_models.ipynb) to custom layers and models.
