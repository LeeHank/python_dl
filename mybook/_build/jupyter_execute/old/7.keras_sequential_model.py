#!/usr/bin/env python
# coding: utf-8

# # The Sequential model

# ## Setup

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ## When to use a Sequential model
# 
# A `Sequential` model is appropriate for **a plain stack of layers**
# where each layer has **exactly one input tensor and one output tensor**.
# 
# Schematically, the following `Sequential` model:

# In[29]:


# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)


# is equivalent to this function:

# In[3]:


# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))


# A Sequential model is **not appropriate** when:
# 
# - Your model has multiple inputs or multiple outputs
# - Any of your layers has multiple inputs or multiple outputs
# - You need to do layer sharing
# - You want non-linear topology (e.g. a residual connection, a multi-branch
# model)

# ## Creating a Sequential model
# 
# You can create a Sequential model by passing a list of layers to the Sequential
# constructor:

# In[4]:


model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)


# * 可以用 `.layers` attribute，取得 model 的所有 layer

# In[5]:


model.layers


# You can also create a Sequential model incrementally via the `add()` method:

# In[6]:


model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))


# Note that there's also a corresponding `pop()` method to remove layers:
# a Sequential model behaves very much like a list of layers.

# In[7]:


model.pop()
print(len(model.layers))  # 2


# Also note that the Sequential constructor accepts a `name` argument, just like
# any layer or model in Keras. This is useful to annotate TensorBoard graphs
# with semantically meaningful names.

# In[8]:


model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))


# ## Specifying the input shape in advance

# Generally, all layers in Keras need to know the shape of their inputs
# in order to be able to create their weights. So when you create a layer like
# this, initially, it has no weights:

# In[10]:


layer = layers.Dense(3)
layer.weights  # Empty


# It creates its weights the first time it is called on an input, since the shape
# of the weights depends on the shape of the inputs:

# In[11]:


# Call layer on a test input
x = tf.ones((1, 4))
y = layer(x)
layer.weights  # Now it has weights, of shape (4, 3) and (3,)


# Naturally, this also applies to Sequential models. When you instantiate a
# Sequential model without an input shape, it isn't "built": it has no weights
# (and calling
# `model.weights` results in an error stating just this). The weights are created
# when the model first sees some input data:

# In[12]:


model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # No weights at this stage!

# At this point, you can't do this:
# model.weights

# You also can't do this:
# model.summary()

# Call the model on a test input
x = tf.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6


# Once a model is "built", you can call its `summary()` method to display its
# contents:

# In[13]:


model.summary()


# However, it can be very useful when building a Sequential model incrementally
# to be able to display the summary of the model so far, including the current
# output shape. In this case, you should start your model by passing an `Input`
# object to your model, so that it knows its input shape from the start:

# In[14]:


model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()


# * 要注意的是，`Input` object 沒有被顯示出來，因為他就不是 layer

# In[15]:


model.layers


# * 一個簡單的 best practice 是，在第一個 layer，標上 `input_shape` 的 設定：

# In[16]:


model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()


# * 這樣做的話，不需要餵入第一筆 tensor data，就 initialize weights 了 (因為他知道 input shape 了)

# ## A common debugging workflow: `add()` + `summary()`
# 
# When building a new Sequential architecture, it's useful to incrementally stack
# layers with `add()` and frequently print model summaries. For instance, this
# enables you to monitor how a stack of `Conv2D` and `MaxPooling2D` layers is
# downsampling image feature maps:

# In[20]:


model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu")) # 高, 寬 的大小：(250-5+1)/2 = 123
model.add(layers.Conv2D(32, 3, activation="relu")) # 高寬的大小： (123-3+1)/1 = 121
model.add(layers.MaxPooling2D(3)) # default的 stride = kernel_size，所以高寬變成 (121-3+1)/3 = 40

# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:
model.summary()


# In[21]:


# The answer was: (40, 40, 32), so we can keep downsampling...
model.add(layers.Conv2D(32, 3, activation="relu")) # (40-3+1)/1 = 38
model.add(layers.Conv2D(32, 3, activation="relu")) # (38-3+1)/1 = 36
model.add(layers.MaxPooling2D(3)) # 36/3 = 12 
model.add(layers.Conv2D(32, 3, activation="relu")) # (12-3+1)/1 = 10
model.add(layers.Conv2D(32, 3, activation="relu")) # (10-3+1)/1 = 8
model.add(layers.MaxPooling2D(2)) # 8/2 = 4

# And now?
model.summary()


# In[22]:


# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D()) # 32

# Finally, we add a classification layer.
model.add(layers.Dense(10))

model.summary()


# Very practical, right?
# 

# ## What to do once you have a model
# 
# Once your model architecture is ready, you will want to:
# 
# - Train your model, evaluate it, and run inference. See our
# [guide to training & evaluation with the built-in loops](
# https://www.tensorflow.org/guide/keras/train_and_evaluate/)
# - Save your model to disk and restore it. See our
# [guide to serialization & saving](https://www.tensorflow.org/guide/keras/save_and_serialize/).
# - Speed up model training by leveraging multiple GPUs. See our
# [guide to multi-GPU and distributed training](https://keras.io/guides/distributed_training/).

# ## Feature extraction with a Sequential model
# 
# Once a Sequential model has been built, it behaves like a [Functional API
# model](https://www.tensorflow.org/guide/keras/functional/). This means that every layer has an `input`
# and `output` attribute. These attributes can be used to do neat things, like
# quickly
# creating a model that extracts the outputs of all intermediate layers in a
# Sequential model:

# In[23]:


initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)


# In[26]:


len(features)


# Here's a similar example that only extract features from one layer:

# In[32]:


initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)


# ## Transfer learning with a Sequential model
# 
# Transfer learning consists of freezing the bottom layers in a model and only training
# the top layers. If you aren't familiar with it, make sure to read our [guide
# to transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning/).
# 
# Here are two common transfer learning blueprint involving Sequential models.
# 
# First, let's say that you have a Sequential model, and you want to freeze all
# layers except the last one. In this case, you would simply iterate over
# `model.layers` and set `layer.trainable = False` on each layer, except the
# last one. Like this:
# 
# ```python
# model = keras.Sequential([
#     keras.Input(shape=(784)),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(10),
# ])
# 
# # Presumably you would want to first load pre-trained weights.
# model.load_weights(...)
# 
# # Freeze all layers except the last one.
# for layer in model.layers[:-1]:
#   layer.trainable = False
# 
# # Recompile and train (this will only update the weights of the last layer).
# model.compile(...)
# model.fit(...)
# ```
# 
# Another common blueprint is to use a Sequential model to stack a pre-trained
# model and some freshly initialized classification layers. Like this:
# 
# ```python
# # Load a convolutional base with pre-trained weights
# base_model = keras.applications.Xception(
#     weights='imagenet',
#     include_top=False,
#     pooling='avg')
# 
# # Freeze the base model
# base_model.trainable = False
# 
# # Use a Sequential model to add a trainable classifier on top
# model = keras.Sequential([
#     base_model,
#     layers.Dense(1000),
# ])
# 
# # Compile & train
# model.compile(...)
# model.fit(...)
# ```
# 
# If you do transfer learning, you will probably find yourself frequently using
# these two patterns.

# That's about all you need to know about Sequential models!
# 
# To find out more about building models in Keras, see:
# 
# - [Guide to the Functional API](https://www.tensorflow.org/guide/keras/functional/)
# - [Guide to making new Layers & Models via subclassing](
# https://www.tensorflow.org/guide/keras/custom_layers_and_models/)
