#!/usr/bin/env python
# coding: utf-8

# # Introduction to graphs and `tf.function`

# ## Overview
# 
# This guide goes beneath the surface of TensorFlow and Keras to demonstrate how TensorFlow works. If you instead want to immediately get started with Keras, check out the [collection of Keras guides](https://www.tensorflow.org/guide/keras/).
# 
# In this guide, you'll learn how TensorFlow allows you to make simple changes to your code to get graphs, how graphs are stored and represented, and how you can use them to accelerate your models.
# 
# Note: For those of you who are only familiar with TensorFlow 1.x, this guide demonstrates a very different view of graphs.
# 
# **This is a big-picture overview that covers how `tf.function` allows you to switch from eager execution to graph execution.** For a more complete specification of `tf.function`, go to the [Better performance with `tf.function`](./function.ipynb) guide.
# 

# ### What are graphs?
# 
# In the previous three guides, you ran TensorFlow **eagerly**. This means TensorFlow operations are executed by Python, operation by operation, and returning results back to Python.
# 
# While eager execution has several unique advantages, graph execution enables portability outside Python and tends to offer better performance. **Graph execution** means that tensor computations are executed as a *TensorFlow graph*, sometimes referred to as a `tf.Graph` or simply a "graph."
# 
# **Graphs are data structures that contain a set of `tf.Operation` objects, which represent units of computation; and `tf.Tensor` objects, which represent the units of data that flow between operations.** They are defined in a `tf.Graph` context. Since these graphs are data structures, they can be saved, run, and restored all without the original Python code.
# 
# This is what a TensorFlow graph representing a two-layer neural network looks like when visualized in TensorBoard:

# <img alt="A simple TensorFlow graph" src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/intro_to_graphs/two-layer-network.png?raw=1">

# ### The benefits of graphs
# 
# With a graph, you have a great deal of flexibility.  You can use your TensorFlow graph in environments that don't have a Python interpreter, like mobile applications, embedded devices, and backend servers. TensorFlow uses graphs as the format for [saved models](./saved_model.ipynb) when it exports them from Python.
# 
# Graphs are also easily optimized, allowing the compiler to do transformations like:
# 
# * Statically infer the value of tensors by folding constant nodes in your computation *("constant folding")*.
# * Separate sub-parts of a computation that are independent and split them between threads or devices.
# * Simplify arithmetic operations by eliminating common subexpressions.
# 

# There is an entire optimization system, [Grappler](./graph_optimization.ipynb), to perform this and other speedups.
# 
# In short, graphs are extremely useful and let your TensorFlow run **fast**, run **in parallel**, and run efficiently **on multiple devices**.
# 
# However, you still want to define your machine learning models (or other computations) in Python for convenience, and then automatically construct graphs when you need them.

# ## Setup

# Import some necessary libraries:

# In[1]:


import tensorflow as tf
import timeit
from datetime import datetime


# ## Taking advantage of graphs
# 
# You create and run a graph in TensorFlow by using `tf.function`, either as a direct call or as a decorator. `tf.function` takes a regular function as input and returns a `Function`. **A `Function` is a Python callable that builds TensorFlow graphs from the Python function. You use a `Function` in the same way as its Python equivalent.**
# 

# In[2]:


# Define a Python function.
def a_regular_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# `a_function_that_uses_a_graph` is a TensorFlow `Function`.
a_function_that_uses_a_graph = tf.function(a_regular_function)

# Make some tensors.
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
# Call a `Function` like a Python function.
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
assert(orig_value == tf_function_value)


# On the outside, a `Function` looks like a regular function you write using TensorFlow operations. [Underneath](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/def_function.py), however, it is *very different*. A `Function` **encapsulates several `tf.Graph`s behind one API** (learn more in the _Polymorphism_ section). That is how a `Function` is able to give you the benefits of graph execution, like speed and deployability (refer to _The benefits of graphs_ above).

# `tf.function` applies to a function *and all other functions it calls*:

# In[3]:


def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Use the decorator to make `outer_function` a `Function`.
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
outer_function(tf.constant([[1.0, 2.0]])).numpy()


# If you have used TensorFlow 1.x, you will notice that at no time did you need to define a `Placeholder` or `tf.Session`.

# ### Converting Python functions to graphs
# 
# Any function you write with TensorFlow will contain a mixture of built-in TF operations and Python logic, such as `if-then` clauses, loops, `break`, `return`, `continue`, and more. While TensorFlow operations are easily captured by a `tf.Graph`, Python-specific logic needs to undergo an extra step in order to become part of the graph. `tf.function` uses a library called AutoGraph (`tf.autograph`) to convert Python code into graph-generating code.
# 

# In[5]:


def simple_relu(x):
  if tf.greater(x, 0):
    return x
  else:
    return 0

# `tf_simple_relu` is a TensorFlow `Function` that wraps `simple_relu`.
tf_simple_relu = tf.function(simple_relu)

print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())


# Though it is unlikely that you will need to view graphs directly, you can inspect the outputs to check the exact results. These are not easy to read, so no need to look too carefully!

# In[6]:


# This is the graph-generating output of AutoGraph.
print(tf.autograph.to_code(simple_relu))


# In[7]:


# This is the graph itself.
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())


# Most of the time, `tf.function` will work without  special considerations. However, there are some caveats, and the [`tf.function` guide](./function.ipynb) can help here, as well as the [complete AutoGraph reference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md).

# ### Polymorphism: one `Function`, many graphs
# 
# A `tf.Graph` is specialized to a specific type of inputs (for example, tensors with a specific [`dtype`](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType) or objects with the same [`id()`](https://docs.python.org/3/library/functions.html#id])).
# 
# Each time you invoke a `Function` with a set of arguments that can't be handled by any of its existing graphs (such as arguments with new `dtypes` or incompatible shapes), `Function` creates a new `tf.Graph` specialized to those new arguments. The type specification of a `tf.Graph`'s inputs is known as its **input signature** or just a **signature**. For more information regarding when a new `tf.Graph` is generated and how that can be controlled, go to the _Rules of tracing_ section of the [Better performance with `tf.function`](./function.ipynb) guide.
# 
# The `Function` stores the `tf.Graph` corresponding to that signature in a `ConcreteFunction`. **A `ConcreteFunction` is a wrapper around a `tf.Graph`.**
# 

# In[8]:


@tf.function
def my_relu(x):
  return tf.maximum(0., x)

# `my_relu` creates new graphs as it observes more signatures.
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))
print(my_relu(tf.constant([3., -3.])))


# If the `Function` has already been called with that signature, `Function` does not create a new `tf.Graph`.

# In[9]:


# These two calls do *not* create new graphs.
print(my_relu(tf.constant(-2.5))) # Signature matches `tf.constant(5.5)`.
print(my_relu(tf.constant([-1., 1.]))) # Signature matches `tf.constant([3., -3.])`.


# Because it's backed by multiple graphs, a `Function` is **polymorphic**. That enables it to support more input types than a single `tf.Graph` could represent, and to optimize each `tf.Graph` for better performance.

# In[10]:


# There are three `ConcreteFunction`s (one for each graph) in `my_relu`.
# The `ConcreteFunction` also knows the return type and shape!
print(my_relu.pretty_printed_concrete_signatures())


# ## Using `tf.function`
# 
# So far, you've learned how to convert a Python function into a graph simply by using `tf.function` as a decorator or wrapper. But in practice, getting `tf.function` to work correctly can be tricky! In the following sections, you'll learn how you can make your code work as expected with `tf.function`.

# ### Graph execution vs. eager execution
# 
# The code in a `Function` can be executed both eagerly and as a graph. By default, `Function` executes its code as a graph:
# 

# In[11]:


@tf.function
def get_MSE(y_true, y_pred):
  sq_diff = tf.pow(y_true - y_pred, 2)
  return tf.reduce_mean(sq_diff)


# In[12]:


y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print(y_true)
print(y_pred)


# In[13]:


get_MSE(y_true, y_pred)


# To verify that your `Function`'s graph is doing the same computation as its equivalent Python function, you can make it execute eagerly with `tf.config.run_functions_eagerly(True)`. This is a switch that **turns off `Function`'s ability to create and run graphs**, instead of executing the code normally.

# In[14]:


tf.config.run_functions_eagerly(True)


# In[15]:


get_MSE(y_true, y_pred)


# In[16]:


# Don't forget to set it back when you are done.
tf.config.run_functions_eagerly(False)


# However, `Function` can behave differently under graph and eager execution. The Python [`print`](https://docs.python.org/3/library/functions.html#print) function is one example of how these two modes differ. Let's check out what happens when you insert a `print` statement to your function and call it repeatedly.

# In[17]:


@tf.function
def get_MSE(y_true, y_pred):
  print("Calculating MSE!")
  sq_diff = tf.pow(y_true - y_pred, 2)
  return tf.reduce_mean(sq_diff)


# Observe what is printed:

# In[18]:


error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)


# Is the output surprising? **`get_MSE` only printed once even though it was called *three* times.**
# 
# To explain, the `print` statement is executed when `Function` runs the original code in order to create the graph in a process known as "tracing" (refer to the _Tracing_ section of the [`tf.function` guide](./function.ipynb). **Tracing captures the TensorFlow operations into a graph, and `print` is not captured in the graph.**  That graph is then executed for all three calls **without ever running the Python code again**.
# 
# As a sanity check, let's turn off graph execution to compare:

# In[19]:


# Now, globally set everything to run eagerly to force eager execution.
tf.config.run_functions_eagerly(True)


# In[20]:


# Observe what is printed below.
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)


# In[21]:


tf.config.run_functions_eagerly(False)


# `print` is a *Python side effect*, and there are other differences that you should be aware of when converting a function into a `Function`. Learn more in the _Limitations_ section of the [Better performance with `tf.function`](./function.ipynb) guide.

# Note: If you would like to print values in both eager and graph execution, use `tf.print` instead.

# ### Non-strict execution
# 
# <a id="non-strict"></a>
# 
# Graph execution only executes the operations necessary to produce the observable effects, which includes:
# 
# - The return value of the function
# - Documented well-known side-effects such as:
#   - Input/output operations, like `tf.print`
#   - Debugging operations, such as the assert functions in `tf.debugging`
#   - Mutations of `tf.Variable`
# 
# This behavior is usually known as "Non-strict execution", and differs from eager execution, which steps through all of the program operations, needed or not.
# 
# In particular, runtime error checking does not count as an observable effect. If an operation is skipped because it is unnecessary, it cannot raise any runtime errors.
# 
# In the following example, the "unnecessary" operation `tf.gather` is skipped during graph execution, so the runtime error `InvalidArgumentError` is not raised as it would be in eager execution. Do not rely on an error being raised while executing a graph.

# In[22]:


def unused_return_eager(x):
  # Get index 1 will fail when `len(x) == 1`
  tf.gather(x, [1]) # unused 
  return x

try:
  print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
  # All operations are run during eager execution so an error is raised.
  print(f'{type(e).__name__}: {e}')


# In[23]:


@tf.function
def unused_return_graph(x):
  tf.gather(x, [1]) # unused
  return x

# Only needed operations are run during graph execution. The error is not raised.
print(unused_return_graph(tf.constant([0.0])))


# ### `tf.function` best practices
# 
# It may take some time to get used to the behavior of `Function`.  To get started quickly, first-time users should play around with decorating toy functions with `@tf.function` to get experience with going from eager to graph execution.
# 
# *Designing for `tf.function`* may be your best bet for writing graph-compatible TensorFlow programs. Here are some tips:
# -  Toggle between eager and graph execution early and often with `tf.config.run_functions_eagerly` to pinpoint if/ when the two modes diverge.
# - Create `tf.Variable`s
# outside the Python function and modify them on the inside. The same goes for objects that use `tf.Variable`, like `tf.keras.layers`, `tf.keras.Model`s and `tf.keras.optimizers`.
# - Avoid writing functions that depend on outer Python variables, excluding `tf.Variable`s and Keras objects. Learn more in _Depending on Python global and free variables_ of the [`tf.function` guide](./function.ipynb).
# - Prefer to write functions which take tensors and other TensorFlow types as input. You can pass in other object types but be careful! Learn more in _Depending on Python objects_ of the [`tf.function` guide](./function.ipynb).
# - Include as much computation as possible under a `tf.function` to maximize the performance gain. For example, decorate a whole training step or the entire training loop.
# 

# ## Seeing the speed-up

# `tf.function` usually improves the performance of your code, but the amount of speed-up depends on the kind of computation you run. Small computations can be dominated by the overhead of calling a graph. You can measure the difference in performance like so:

# In[24]:


x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)

def power(x, y):
  result = tf.eye(10, dtype=tf.dtypes.int32)
  for _ in range(y):
    result = tf.matmul(x, result)
  return result


# In[25]:


print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000))


# In[26]:


power_as_graph = tf.function(power)
print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000))


# `tf.function` is commonly used to speed up training loops, and you can learn more about it in the _Speeding-up your training step with `tf.function`_ section of the [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) with Keras guide.
# 
# Note: You can also try `tf.function(jit_compile=True)` for a more significant performance boost, especially if your code is heavy on TensorFlow control flow and uses many small tensors. Learn more in the _Explicit compilation with `tf.function(jit_compile=True)`_ section of the [XLA overview](https://www.tensorflow.org/xla).

# ### Performance and trade-offs
# 
# Graphs can speed up your code, but the process of creating them has some overhead. For some functions, the creation of the graph takes more time than the execution of the graph. **This investment is usually quickly paid back with the performance boost of subsequent executions, but it's important to be aware that the first few  steps of any large model training can be slower due to tracing.**
# 
# No matter how large your model, you want to avoid tracing frequently. The [`tf.function` guide](./function.ipynb) discusses how to set input specifications and use tensor arguments to avoid retracing in the _Controlling retracing_ section. If you find you are getting unusually poor performance, it's a good idea to check if you are retracing accidentally.

# ## When is a `Function` tracing?
# 
# To figure out when your `Function` is tracing, add a `print` statement to its code. As a rule of thumb, `Function` will execute the `print` statement every time it traces.

# In[27]:


@tf.function
def a_function_with_python_side_effect(x):
  print("Tracing!") # An eager-only side effect.
  return x * x + tf.constant(2)

# This is traced the first time.
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect.
print(a_function_with_python_side_effect(tf.constant(3)))


# In[28]:


# This retraces each time the Python argument changes,
# as a Python argument could be an epoch count or other
# hyperparameter.
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))


# New Python arguments always trigger the creation of a new graph, hence the extra tracing.
# 

# ## Next steps
# 
# You can learn more about `tf.function` on the API reference page and by following the [Better performance with `tf.function`](./function.ipynb) guide.
