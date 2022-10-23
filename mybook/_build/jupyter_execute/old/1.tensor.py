#!/usr/bin/env python
# coding: utf-8

# # Introduction to Tensors

# In[2]:


import tensorflow as tf
import numpy as np


# * Tensors 就是 multi-dimensional arrays + uniform type (called a `dtype`), 所以就像 `numpy` 的 `np.arrays`. 
# * 但差別是，所有的 tensors 都是 immutable，所以 you can never update the contents of a tensor, only create a new one.

# ## Basics

# * 首先，建立一些 basic tensors
# * Here is a "scalar" or "rank-0" tensor . 
# * A scalar contains a single value, and no "axes".

# * 複習一下 numpy 學果的 shape, axes, 和 rank  
#   * `[2,3,4]` 這種 array，他的 shape 是 (3,)，只有 1 個軸 (第一軸)，所以 rank = 1
#   * `[[1,2,3], [4,5,6]]` 這種 array，他的 shape 是 (2, 3)，有兩個軸 (第一軸和第二軸)，所以 rank = 2  
#   * `[[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]` 這種 array，他的 shape 是 (2,2,3)，所以 rank = 3
#   * `2` 這種 array，他的 shape 是 ()，根本沒有軸，所以他的 rank = 0 
# * 接下來，對 tensor 的介紹，概念完全一樣：

# ### Scalar (rank-0 tensor)

# In[4]:


# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)


# * 對 tensor 最重要的 attribute，就是 `shape` 和 `dtype`，可以看到他分別 print 出這兩個 attribute 的結果：
#   * `shape = ()`，因為從 list 的角度，他根本不是 list，所以連第一軸都沒有，rank = 0
#   * `dtype = int32` 表示是整數型 type 

# ### Vector (rank-1 tensor)

# In[5]:


# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)


# ### Matrix (rank-2 tensor)

# In[6]:


# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)


# <table>
# <tr>
#   <th>A scalar, shape: <code>[]</code></th>
#   <th>A vector, shape: <code>[3]</code></th>
#   <th>A matrix, shape: <code>[3, 2]</code></th>
# </tr>
# <tr>
#   <td>
#    <img src="images/tensor/scalar.png" alt="A scalar, the number 4" />
#   </td>
# 
#   <td>
#    <img src="images/tensor/vector.png" alt="The line with 3 sections, each one containing a number."/>
#   </td>
#   <td>
#    <img src="images/tensor/matrix.png" alt="A 3x2 grid, with each cell containing a number.">
#   </td>
# </tr>
# </table>
# 

# ### More axes tensor

# * 之後做訓練的時候，都會是多軸的 tensor
# * 來看看以下的 3 軸 範例

# In[7]:


# There can be an arbitrary number of
# axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)


# There are many ways you might visualize a tensor with more than two axes.

# <table>
# <tr>
#   <th colspan=3>A 3-axis tensor, shape: <code>[3, 2, 5]</code></th>
# <tr>
# <tr>
#   <td>
#    <img src="images/tensor/3-axis_numpy.png"/>
#   </td>
#   <td>
#    <img src="images/tensor/3-axis_front.png"/>
#   </td>
# 
#   <td>
#    <img src="images/tensor/3-axis_block.png"/>
#   </td>
# </tr>
# 
# </table>

# ### tensor <-> numpy

# * 用 `np.array(tensor_obj)` 或 `tensor_obj.numpy()` 來將 tensor 轉成 numpy
# * 用 `convert_to_tensor(numpy_obj)` 將 numpy 轉成 tensor

# In[13]:


rank_2_tensor


# In[8]:


np.array(rank_2_tensor)


# In[9]:


rank_2_tensor.numpy()


# In[14]:


numpy_obj = np.array([[1,2], [3,4], [5,6]])
numpy_obj


# In[15]:


tf.convert_to_tensor(numpy_obj)


# ### type

# Tensors often contain floats and ints, but have many other types, including:
# 
# * complex numbers
# * strings

# ### shape

# * The base `tf.Tensor` class requires tensors to be "rectangular"---that is, along each axis, every element is the same size.  
# * However, there are specialized types of tensors that can handle different shapes:
#   * Ragged tensors (see [RaggedTensor](#ragged_tensors) below)
#   * Sparse tensors (see [SparseTensor](#sparse_tensors) below)

# ### operations

# * 基本加減乘除：
#   * `+` 或 `tf.add()`
#   * `-`
#   * `*` 或 `tf.multiply()` (element-wise 相乘)
# * 矩陣運算
#   * `@` 或 `tf.matmul()` (矩陣相乘)
# * summarise:  
#   * `reduce_max()`. 
#   * `tf.math.argmax()`. 

# In[16]:


a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")


# In[17]:


print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication


# Tensors are used in all kinds of operations (or "Ops").

# In[12]:


c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.math.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))


# Note: Typically, anywhere a TensorFlow function expects a `Tensor` as input, the function will also accept anything that can be converted to a `Tensor` using `tf.convert_to_tensor`. See below for an example.

# In[13]:


tf.reduce_max([1,2,3])


# In[14]:


tf.reduce_max(np.array([1,2,3]))


# ## About shapes

# 一些名詞：
# 
# * **Shape**: The length (number of elements) of each of the axes of a tensor.
# * **Rank**: Number of tensor axes.  A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
# * **Axis** or **Dimension**: A particular dimension of a tensor.
# * **Size**: The total number of items in the tensor, the product of the shape vector's elements.
# 

# Note: Although you may see reference to a "tensor of two dimensions", a rank-2 tensor does not usually describe a 2D space.

# Tensors and `tf.TensorShape` objects have convenient properties for accessing these:

# In[18]:


rank_4_tensor = tf.zeros([3, 2, 4, 5])


# <table>
# <tr>
#   <th colspan=2>A rank-4 tensor, shape: <code>[3, 2, 4, 5]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/shape.png" alt="A tensor shape is like a vector.">
#     <td>
# <img src="images/tensor/4-axis_block.png" alt="A 4-axis tensor">
#   </td>
#   </tr>
# </table>
# 

# In[19]:


print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())


# But note that the `Tensor.ndim` and `Tensor.shape` attributes don't return `Tensor` objects. If you need a `Tensor` use the `tf.rank` or `tf.shape` function. This difference is subtle, but it can be important when building graphs (later).

# In[20]:


tf.rank(rank_4_tensor)


# In[21]:


tf.shape(rank_4_tensor)


# While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.
# 
# <table>
# <tr>
# <th>Typical axis order</th>
# </tr>
# <tr>
#     <td>
# <img src="images/tensor/shape2.png" alt="Keep track of what each axis is. A 4-axis tensor might be: Batch, Width, Height, Features">
#   </td>
# </tr>
# </table>

# ## Indexing

# ### Single-axis indexing
# 
# TensorFlow follows standard Python indexing rules, similar to [indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings){:.external}, and the basic rules for NumPy indexing.
# 
# * indexes start at `0`
# * negative indices count backwards from the end
# * colons, `:`, are used for slices: `start:stop:step`
# 

# In[22]:


rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())


# Indexing with a scalar removes the axis:

# In[23]:


print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())


# Indexing with a `:` slice keeps the axis:

# In[24]:


print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())


# ### Multi-axis indexing

# Higher rank tensors are indexed by passing multiple indices.
# 
# The exact same rules as in the single-axis case apply to each axis independently.

# In[25]:


print(rank_2_tensor.numpy())


# Passing an integer for each index, the result is a scalar.

# In[26]:


# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())


# You can index using any combination of integers and slices:

# In[27]:


# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")


# Here is an example with a 3-axis tensor:

# In[28]:


print(rank_3_tensor[:, :, 4])


# <table>
# <tr>
# <th colspan=2>Selecting the last feature across all locations in each example in the batch </th>
# </tr>
# <tr>
#     <td>
# <img src="images/tensor/index1.png" alt="A 3x2x5 tensor with all the values at the index-4 of the last axis selected.">
#   </td>
#       <td>
# <img src="images/tensor/index2.png" alt="The selected values packed into a 2-axis tensor.">
#   </td>
# </tr>
# </table>

# Read the [tensor slicing guide](https://tensorflow.org/guide/tensor_slicing) to learn how you can apply indexing to manipulate individual elements in your tensors.

# ## Manipulating Shapes
# 
# Reshaping a tensor is of great utility. 
# 

# In[35]:


# Shape returns a `TensorShape` object that shows the size along each axis
x = tf.constant([[1], [2], [3]])
print(x)
print(x.shape)


# In[36]:


# You can convert this object into a Python list, too
print(x.shape.as_list())


# You can reshape a tensor into a new shape. The `tf.reshape` operation is fast and cheap as the underlying data does not need to be duplicated.

# In[31]:


# You can reshape a tensor to a new shape.
# Note that you're passing in a list
reshaped = tf.reshape(x, [1, 3])


# In[37]:


print(x)
print(x.shape)
print(reshaped)
print(reshaped.shape)


# * The data maintains its layout in memory and a new tensor is created, with the requested shape, pointing to the same data. 
# * TensorFlow uses C-style "row-major" memory ordering, where incrementing the rightmost index corresponds to a single step in memory.

# In[38]:


print(rank_3_tensor)


# If you flatten a tensor you can see what order it is laid out in memory.

# In[39]:


# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))


# Typically the only reasonable use of `tf.reshape` is to combine or split adjacent axes (or add/remove `1`s).
# 
# For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do, as the slices do not mix:

# In[41]:


print(rank_3_tensor, "\n")
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))


# <table>
# <th colspan=3>
# Some good reshapes.
# </th>
# <tr>
#   <td>
# <img src="images/tensor/reshape-before.png" alt="A 3x2x5 tensor">
#   </td>
#   <td>
#   <img src="images/tensor/reshape-good1.png" alt="The same data reshaped to (3x2)x5">
#   </td>
#   <td>
# <img src="images/tensor/reshape-good2.png" alt="The same data reshaped to 3x(2x5)">
#   </td>
# </tr>
# </table>
# 

# Reshaping will "work" for any new shape with the same total number of elements, but it will not do anything useful if you do not respect the order of the axes.
# 
# Swapping axes in `tf.reshape` does not work; you need `tf.transpose` for that. 
# 

# In[33]:


# Bad examples: don't do this

# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")


# <table>
# <th colspan=3>
# Some bad reshapes.
# </th>
# <tr>
#   <td>
# <img src="images/tensor/reshape-bad.png" alt="You can't reorder axes, use tf.transpose for that">
#   </td>
#   <td>
# <img src="images/tensor/reshape-bad4.png" alt="Anything that mixes the slices of data together is probably wrong.">
#   </td>
#   <td>
# <img src="images/tensor/reshape-bad2.png" alt="The new shape must fit exactly.">
#   </td>
# </tr>
# </table>

# You may run across not-fully-specified shapes. Either the shape contains a `None` (an axis-length is unknown) or the whole shape is `None` (the rank of the tensor is unknown).
# 
# Except for [tf.RaggedTensor](#ragged_tensors), such shapes will only occur in the context of TensorFlow's symbolic, graph-building  APIs:
# 
# * [tf.function](function.ipynb) 
# * The [keras functional API](https://www.tensorflow.org/guide/keras/functional).
# 

# ## More on `DTypes`

# * 我們可以用 `Tensor.dtype` 來看 data type
# * 當我們用建立 tensor 時，可以
#   * 不指定 dtype, 那tensorflow 會自動幫你挑適合的 (e.g. `tf.int32`, `tf.float32`)  
#   * 直接指定 dtype  
#   * 也可以用 `tf.cast()` 來轉換 dtype

# In[43]:


# 不指定 dtype
tt = tf.constant([2.2, 3.3, 4.4])
print(tt.dtype)

# 指定 dtype
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
print(the_f64_tensor.dtype)

# cast type
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
print(the_f16_tensor.dtype)

# Now, cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
print(the_u8_tensor.dtype)


# ## Broadcasting

# * Broadcasting is a concept borrowed from the [equivalent feature in NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html){:.external}.  
# * In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.
# * The simplest and most common case is when you attempt to multiply or add a tensor to a scalar.  In that case, the scalar is broadcast to be the same shape as the other argument. 

# In[44]:


x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)


# Likewise, axes with length 1 can be stretched out to match the other arguments.  Both arguments can be stretched in the same computation.
# 
# In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to produce a 3x4 matrix. Note how the leading 1 is optional: The shape of y is `[4]`.

# In[45]:


# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))


# <table>
# <tr>
#   <th>A broadcasted add: a <code>[3, 1]</code> times a <code>[1, 4]</code> gives a <code>[3,4]</code> </th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/broadcasting.png" alt="Adding a 3x1 matrix to a 4x1 matrix results in a 3x4 matrix">
#   </td>
# </tr>
# </table>
# 

# Here is the same operation without broadcasting:

# In[46]:


x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading


# Most of the time, broadcasting is both time and space efficient, as the broadcast operation never materializes the expanded tensors in memory.  
# 
# You see what broadcasting looks like using `tf.broadcast_to`.

# In[47]:


print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))


# Unlike a mathematical op, for example, `broadcast_to` does nothing special to save memory.  Here, you are materializing the tensor.
# 
# It can get even more complicated.  [This section](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html){:.external} of Jake VanderPlas's book _Python Data Science Handbook_ shows more broadcasting tricks (again in NumPy).

# ## Ragged Tensors

# * A tensor with variable numbers of elements along some axis is called "ragged". Use `tf.ragged.RaggedTensor` for ragged data.  
# * For example, This cannot be represented as a regular tensor:

# <table>
# <tr>
#   <th>A `tf.RaggedTensor`, shape: <code>[4, None]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/ragged.png" alt="A 2-axis ragged tensor, each row can have a different length.">
#   </td>
# </tr>
# </table>

# In[48]:


ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]


# In[49]:


try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")


# Instead create a `tf.RaggedTensor` using `tf.ragged.constant`:

# In[50]:


ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)


# The shape of a `tf.RaggedTensor` will contain some axes with unknown lengths:

# In[51]:


print(ragged_tensor.shape)


# ## String tensors
# 
# `tf.string` is a `dtype`, which is to say you can represent data as strings (variable-length byte arrays) in tensors.
# 
# The strings are atomic and cannot be indexed the way Python strings are. The length of the string is not one of the axes of the tensor. See `tf.strings` for functions to manipulate them.

# Here is a scalar string tensor:

# In[52]:


# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)


# And a vector of strings:

# <table>
# <tr>
#   <th>A vector of strings, shape: <code>[3,]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/strings.png" alt="The string length is not one of the tensor's axes.">
#   </td>
# </tr>
# </table>

# In[53]:


# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
print(tensor_of_strings)


# In the above printout the `b` prefix indicates that `tf.string` dtype is not a unicode string, but a byte-string. See the [Unicode Tutorial](https://www.tensorflow.org/tutorials/load_data/unicode) for more about working with unicode text in TensorFlow.

# If you pass unicode characters they are utf-8 encoded.

# In[54]:


tf.constant("🥳👍")


# Some basic functions with strings can be found in `tf.strings`, including `tf.strings.split`.

# In[55]:


# You can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))


# In[47]:


# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))


# <table>
# <tr>
#   <th>Three strings split, shape: <code>[3, None]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/string-split.png" alt="Splitting multiple strings returns a tf.RaggedTensor">
#   </td>
# </tr>
# </table>

# And `tf.string.to_number`:

# In[56]:


text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))


# Although you can't use `tf.cast` to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.

# In[57]:


byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)


# In[58]:


# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)


# The `tf.string` dtype is used for all raw bytes data in TensorFlow. The `tf.io` module contains functions for converting data to and from bytes, including decoding images and parsing csv.

# ## Sparse tensors
# 
# Sometimes, your data is sparse, like a very wide embedding space.  TensorFlow supports `tf.sparse.SparseTensor` and related operations to store sparse data efficiently.

# <table>
# <tr>
#   <th>A `tf.SparseTensor`, shape: <code>[3, 4]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/sparse.png" alt="An 3x4 grid, with values in only two of the cells.">
#   </td>
# </tr>
# </table>

# In[59]:


# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))


# In[ ]:




