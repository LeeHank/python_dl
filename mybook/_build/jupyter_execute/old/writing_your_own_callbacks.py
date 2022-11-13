#!/usr/bin/env python
# coding: utf-8

# # Writing your own callbacks

# * keras 在 training, evaluation, 和 inference 的時候，會對應到 `model.fit()`, `model.evaluate()` 和 `model.predict()` 這三個高階的 function. 
# * 這三個高階的 function，運行過程中都幫你做了很多事，而 `callbacks`，就是用來客製化這三個 function 進行過程中的行為
# * callback 可以介入的時間點包括：
#   * Global methods:
#     * on_(train|test|predict)_begin(self, logs=None): Called at the beginning of fit/evaluate/predict。例如：
#     * on_(train|test|predict)_end(self, logs=None): Called at the end of fit/evaluate/predict。例如：
#   * Batch-level methods for training/testing/predicting
#     * on_(train|test|predict)_batch_begin(self, batch, logs=None): Called right before processing a batch during training/testing/predicting。
#     * on_(train|test|predict)_batch_end(self, batch, logs=None): Called at the end of training/testing/predicting a batch. Within this method, logs is a dict containing the metrics results.
#   * Epoch-level methods (training only)
#     * on_epoch_begin(self, epoch, logs=None): Called at the beginning of an epoch during training.
#     * on_epoch_end(self, epoch, logs=None): Called at the end of an epoch during training.

# ## setting

# In[1]:


# library
import tensorflow as tf
from tensorflow import keras

# Question: MNIST 分類問題

# model
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_dim=784))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

# data & preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Limit the data to 1000 samples
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# model
model = get_model()


# ## overview

# ### training

# In[10]:


class CustomTrainingCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"Starting training; got log keys: {keys}")

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"Start epoch {epoch} of training; got log keys: {keys}")

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Training: start of batch {batch}; got log keys: {keys}")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Training: end of batch {batch}; got log keys: {keys}")
    
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"End epoch {epoch} of training; got log keys: {keys}")
    
    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print(f"Stop training; got log keys: {keys}")


# * 從上面的定義，可以看到，他把 training 會經過的各個階段都定義了：  
#   * training 開始前，我們可以做一些事 (e.g. 這邊就只是列出，這個階段的 logs 字典，裡面有哪些 key)
#   * epoch 開始前，我們可以做哪些事
#   * 每個 batch 開始前，我們可以做哪些事  
#   * 每個 batch 結束時，系統會預設幫你算 training loss/metric，所以這時可以看到 log 的 key，就是你有定義的 loss/metric
#   * epoch 結束時，系統也會預設幫你算整個 epoch 下來的 training loss/metric，以及 evaluate 在 validation 上的 loss/metric。 所以這時的 log 的 key，也應該看到這些 loss/metric/val_loss/val_metric 
#   * training 結束時，會把各個 epoch 結束時結算的 loss/metric/val_loss/val_metric 一起收起來。

# In[11]:


history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomTrainingCallback()],
)


# * 和前面想的一樣  
# * 再看一下 history 物件，就可以看到，這就是 stop training 時的最終 log

# In[6]:


history.history


# ### Evaluation

# In[7]:


class CustomTestingCallback(keras.callbacks.Callback):    

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
    
    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))


# In[13]:


res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomTestingCallback()]
)


# ### prediction

# In[14]:


class CustomPredictCallback(keras.callbacks.Callback):    
    
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))    

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
    
    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))


# In[16]:


res = model.predict(x_test, batch_size=128, callbacks=[CustomPredictCallback()])


# ## `logs` dictionary

# * 如上面所見， logs 就是個 dictionary，他會在：  
#   * 每個 batch 結束時，紀錄 training 的 loss/metrics
#   * 每個 epoch 結束時，紀錄 training 和 validation 的 loss/metrics

# ## `self.model` attribute

# In[ ]:





# ## How to

# ### 模仿 fit 的 verbose 行為

# * 想做的就是：
#   * 每個 batch 結束 print loss
#   * 每個 epoch 結束，print training & validation 的 loss/metric

# In[29]:


class MimicFitVerbose(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Start epoch {epoch} of training")
    
    def on_train_batch_end(self, batch, logs=None):
        print(
            f"Up to batch {batch}, the average loss is {logs['loss']}"
        )    

    def on_test_batch_end(self, batch, logs=None):
        print(
            f"Up to batch {batch}, the average loss is {logs['loss']}"
        )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


# In[30]:


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    callbacks=[MimicFitVerbose()],
)


# In[ ]:




