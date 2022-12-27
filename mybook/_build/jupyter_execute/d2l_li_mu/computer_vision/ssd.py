#!/usr/bin/env python
# coding: utf-8

# # SSD (single shot)

# * 這一章會用到前幾章的先備知識，來自己寫一個簡單的 ssd  
#   * bounding box
#   * anchor box. 
#   * multiscale-object detection. 
#   * banana dataset. 
# * ssd 簡單快速且被廣泛使用，儘管他只是 object detection 的其中一種模型，但學到的這些概念也都可以類化到其他模型

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# ## Model

# ![单发多框检测模型主要由一个基础网络块和若干多尺度特征块串联而成。](../img/ssd.svg)

# * 上圖描述了 SSD 的 model 設計

# 上圖描述了单发多框检测模型的设计。
# 此模型主要由基础网络组成，其后是几个多尺度特征块。
# 基本网络用于从输入图像中提取特征，因此它可以使用深度卷积神经网络。
# 单发多框检测论文中选用了在分类层之前截断的VGG :cite:`Liu.Anguelov.Erhan.ea.2016`，现在也常用ResNet替代。
# 我们可以设计基础网络，使它输出的高和宽较大。
# 这样一来，基于该特征图生成的锚框数量较多，可以用来检测尺寸较小的目标。
# 接下来的每个多尺度特征块将上一层提供的特征图的高和宽缩小（如减半），并使特征图中每个单元在输入图像上的感受野变得更广阔。
# 
# 回想一下在 :numref:`sec_multiscale-object-detection`中，通过深度神经网络分层表示图像的多尺度目标检测的设计。
# 由于接近 :numref:`fig_ssd`顶部的多尺度特征图较小，但具有较大的感受野，它们适合检测较少但较大的物体。
# 简而言之，通过多尺度特征块，单发多框检测生成不同大小的锚框，并通过预测边界框的类别和偏移量来检测大小不同的目标，因此这是一个多尺度目标检测模型。
# 
# 
# :label:`fig_ssd`
# 
# 在下面，我们将介绍 :numref:`fig_ssd`中不同块的实施细节。
# 首先，我们将讨论如何实施类别和边界框预测。

# ### 類別預測層

# * 假設 object detection 的 object 總數量為 $q$，那總類別就會是 $q+1$，因為會多放一個背景類 0 進去
# * 在某個 image scale 下 (就是某種 img size 下)，假設 feature map 的高寬分別為 $h$ 和 $w$
# * 在這張 feature map 下，以每個 pixel 為中心，生成 $a$ 個 anchor box ($a$ = anchor box 縮寫)，就需要進行 $hwa$ 個 anchor box 的 classification. 
# * 每一次的 anchor box 預測，都需要輸出 $q+1$ 個 pred_prob 
# * 所以，如果使用以往常用的 fully connected layer 來當預測層，這個 layer 就會是拉直成 $h \times w \times a \times (q+1)$ 維的向量，然後才做預測。
# * 很明顯，這個 fully connected layer 會使得參數變過多
# * 所以，這邊借用 NiN (Network in Network) 時學到的概念，我們不要用 fully connected layer 來當 class prediction layer，而是用 convolution layer:
#   * 此 conv layer 的輸出，會和輸入的feature map 高寬相同，都是 h, w，那等於 pixedl-wise 一一對應. 
#   * 通道數就會是 anchor 的數量 乘上 類別的數量 = $a \times (q+1)$  
#   * 總體來說，就是建立了 (h, w, $q \times (q+1)$) 的 convolution layer，該通道就負責該 pixel 對應的所有 anchor box 的預測結果. 
#   * 該 pixecl 對應的通道的向量叫 v，那我們要取該 pixel 所對應的第 i 個 anchor box 的分類結果，就利用 index 去取 
#   * 寫成 code 就長下面這樣

# In[6]:


def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    args:  
      - num_inputs: 前一層 conv layer 的 通道數
      - num_anchors: 做預測時，每個 pixel 會生出幾個 anchor box. 
      - num_classes: 總共有多少個類別要做預測
    return:
      - conv layer
    """
    out = nn.Conv2d(
        num_inputs, 
        num_anchors * (num_classes + 1),
        kernel_size=3, padding=1 # same padding
    )
    
    return out


# * 舉例來說  
#   * 上一層的 feature map 如果 高寬是 4, 通道數是 512 的 conv layer (此時 shape 為 (batch_size, 512, 4, 4)  
#   * 我要對每個 pixel 生出 5 個 anchor box (所以會生出 4x4x5 個 anchor box). 
#   * 對每個 anchor box，做 2 類 (貓 vs 狗) 的預測. 
#   * 那我的 classification layer 會長這樣：  

# In[10]:


# fake feature map
feature_map = torch.zeros((2, 512, 4, 4)) # batch size = 2, 通道數 512, 高寬都是 4

# classifcation layer
cls_conv = cls_predictor(num_inputs = 512, num_anchors = 5, num_classes = 2)

# predict
y1 = cls_conv(feature_map)

# print shape
y1.shape


# * 可以看到，batch size 為 2，然後通道數 15, 高寬仍然是 4. 
# * 這個通道數 = 15，就是 5 個 anchor box 乘上 (2+1) 個類別的預測結果 (+1 是背景類)

# ### bounding box 預測層

# * 除了預測每個 anchor box 所 crop 下來的圖，他的類別外，我們還要預測這個 anchor box 與 真實 bounding box 的 4 個 offset (分別是對 cx, cy, width ,height 的偏移量). 
# * 所以，做法同上，code 寫成：

# In[11]:


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# * 一樣來試試效果

# In[12]:


# fake feature map
feature_map = torch.zeros((2, 512, 4, 4)) # batch size = 2, 通道數 512, 高寬都是 4

# bbox offset prediction layer
bbox_pred_conv = bbox_predictor(num_inputs = 512, num_anchors = 5)

# predict
y1 = bbox_pred_conv(feature_map)

# print shape
y1.shape


# * 從通道數可以證實，20 = 5(個anchor box) x 4 (個 offset 預測值)

# ### 連結到 multi-scale prediction

# * 因為 SSD 會一路將原圖轉出來的 feature map，一路 down-sampling 下去成好幾個 feature map，並對每個 feature map 做上面的預測. 
# * 所以，可能第一個 feature map 的寬高，和第二個 feature map 的寬高就不同了，那預測出來的 conv layer 當然 shape 也不同，就不方便把結果統合再一起，去算 loss
# * 舉例來說：

# In[13]:


# fake feature maps
first_scale_feature_map = torch.zeros((2, 8, 20, 20)) # 這個 feature map，他的寬高是 20
second_scale_feature_map = torch.zeros((2, 16, 10, 10)) # 這個 feature map ，他的寬高已經減半成 20

# 對這兩張 feature maps 的預測
first_cls_conv = cls_predictor(8, 5, 10) # 對第一張 feature map，他 input 的 channel 是 8，然後對每個 pixel 我想生 5 個 anchor box
second_cls_conv = cls_predictor(16, 3, 10) # 對第二張 feature map，他 input 的 channel 是 16，然後對每個 pixel 我想生 3 個 anchor box

# prediction
y1 = first_cls_conv(first_scale_feature_map)
y2 = second_cls_conv(second_scale_feature_map)

# shape
print(y1.shape)
print(y2.shape)


# * 可以看到，這兩個預測結果，除了 batch size 這一維外，其他都長不一樣。
#   * 第一張 feature map，最後的通道數 55，因為 5x(10+1)
#   * 第二張 feature map，最後的通道數 33，因為 3x(10+1)
#   * 且最後的高寬也都不同，一個是 20x20, 一個是 10x10. 
# * 所以，解決的辦法是我只保留第一維(batch_size)，其他三維，先把順序調整成(h,w,channel)再flatten，那就可以 concat 起來了  
# * 至於要先把順序調成 (h, w, channel) 才 flatten，是因為這樣 flatten 後，他的排列會是 該h該w 下 (i.e. 該空間座標(x,y)下) 的預測結果 (anchor box1 的 q+1 個 class 的 pred_prob, anchor box2 的 q+1 個 class 的 pred_prob, ...)

# In[16]:


def flatten_pred(pred):
    reorder_res = pred.permute(0, 2, 3, 1)
    flatten_res = torch.flatten(reorder_res, start_dim = 1) # 從第 1 維開始 flatten(i.e. 保留第 0 維的 batch size)
    return flatten_res


# In[15]:


y1_flatten = flatten_pred(y1)
y2_flatten = flatten_pred(y2)
print(y1_flatten.shape)
print(y2_flatten.shape)


# In[17]:


def concat_preds(preds):
    flatten_list = [flatten_pred(p) for p in preds]
    concat_res = torch.cat(flatten_list, dim=1)
    return concat_res


# In[19]:


concat_preds([y1, y2]).shape


# ### 高和寬減半 block

# * 在 multi-scale detection 的時候，他的 multi-scale 其實就是把原圖經 feature extraction (e.g. 用 resnet50 做 feature extraction) 的 feature map，一路高寬減半下去得到越來越小的 feature map，並對每個 feature map 做預測
# * 那這個高寬減半的過程，當然不是只做個 max-pooling 而已，而是會連續做兩個 (conv-batchnorm-relu) 再 max-pooling 來讓他高寬減半 (這邊的設計就隨意啦，這樣的連續兩個 conv-batchnorm-relu 的設計，只是參考 vgg block 的設計)
# * 這句看不懂：对于此高和宽减半块的输入和输出特征图，因为 1x2 + (3-1) + (3-1) = 6，所以输出中的每个单元在输入上都有一个 6x6 的感受野。因此，高和宽减半块会扩大每个单元在其输出特征图中的感受野。
# * 所以，這邊定義一個這種 block，後續就可以一直再利用：

# In[20]:


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(
            nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=3, 
                padding=1
            )
        )
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# * 來測試一下，如果前一層的 feature map 的 shape 是 (2, 3, 20, 20) # batch_size = 2, 通道數 3, 高寬 20x20. 
# * 那經過高寬減半block，shape 應該要變成 (2, out_channel, 10, 10)

# In[21]:


def forward(x, block):
    return block(x)

forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape


# ### [**基本网络块**]
# 
# 基本网络块用于从输入图像中抽取特征。
# 为了计算简洁，我们构造了一个小的基础网络，该网络串联3个高和宽减半块，并逐步将通道数翻倍。
# 给定输入图像的形状为$256\times256$，此基本网络块输出的特征图形状为$32 \times 32$（$256/2^3=32$）。
# 

# In[8]:


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape


# ### 完整的模型
# 
# [**完整的单发多框检测模型由五个模块组成**]。每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量。在这五个模块中，第一个是基本网络块，第二个到第四个是高和宽减半块，最后一个模块使用全局最大池将高度和宽度都降到1。从技术上讲，第二到第五个区块都是 :numref:`fig_ssd`中的多尺度特征块。
# 

# In[9]:


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


# 现在我们[**为每个块定义前向传播**]。与图像分类任务不同，此处的输出包括：CNN特征图`Y`；在当前尺度下根据`Y`生成的锚框；预测的这些锚框的类别和偏移量（基于`Y`）。
# 

# In[10]:


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


# 回想一下，在 :numref:`fig_ssd`中，一个较接近顶部的多尺度特征块是用于检测较大目标的，因此需要生成更大的锚框。
# 在上面的前向传播中，在每个多尺度特征块上，我们通过调用的`multibox_prior`函数（见 :numref:`sec_anchor`）的`sizes`参数传递两个比例值的列表。
# 在下面，0.2和1.05之间的区间被均匀分成五个部分，以确定五个模块的在不同尺度下的较小值：0.2、0.37、0.54、0.71和0.88。
# 之后，他们较大的值由$\sqrt{0.2 \times 0.37} = 0.272$、$\sqrt{0.37 \times 0.54} = 0.447$等给出。
# 
# [~~超参数~~]
# 

# In[11]:


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


# 现在，我们就可以按如下方式[**定义完整的模型**]`TinySSD`了。
# 

# In[12]:


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 我们[**创建一个模型实例，然后使用它**]对一个$256 \times 256$像素的小批量图像`X`(**执行前向传播**)。
# 
# 如本节前面部分所示，第一个模块输出特征图的形状为$32 \times 32$。
# 回想一下，第二到第四个模块为高和宽减半块，第五个模块为全局汇聚层。
# 由于以特征图的每个单元为中心有$4$个锚框生成，因此在所有五个尺度下，每个图像总共生成$(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$个锚框。
# 

# In[13]:


net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)


# ## 训练模型
# 
# 现在，我们将描述如何训练用于目标检测的单发多框检测模型。
# 
# ### 读取数据集和初始化
# 
# 首先，让我们[**读取**] :numref:`sec_object-detection-dataset`中描述的(**香蕉检测数据集**)。
# 

# In[14]:


batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)


# 香蕉检测数据集中，目标的类别数为1。
# 定义好模型后，我们需要(**初始化其参数并定义优化算法**)。
# 

# In[15]:


device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)


# ### [**定义损失函数和评价函数**]
# 
# 目标检测有两种类型的损失。
# 第一种有关锚框类别的损失：我们可以简单地复用之前图像分类问题里一直使用的交叉熵损失函数来计算；
# 第二种有关正类锚框偏移量的损失：预测偏移量是一个回归问题。
# 但是，对于这个回归问题，我们在这里不使用 :numref:`subsec_normal_distribution_and_squared_loss`中描述的平方损失，而是使用$L_1$范数损失，即预测值和真实值之差的绝对值。
# 掩码变量`bbox_masks`令负类锚框和填充锚框不参与损失的计算。
# 最后，我们将锚框类别和偏移量的损失相加，以获得模型的最终损失函数。
# 

# In[16]:


cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


# 我们可以沿用准确率评价分类结果。
# 由于偏移量使用了$L_1$范数损失，我们使用*平均绝对误差*来评价边界框的预测结果。这些预测结果是从生成的锚框及其预测偏移量中获得的。
# 

# In[17]:


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


# ### [**训练模型**]
# 
# 在训练模型时，我们需要在模型的前向传播过程中生成多尺度锚框（`anchors`），并预测其类别（`cls_preds`）和偏移量（`bbox_preds`）。
# 然后，我们根据标签信息`Y`为生成的锚框标记类别（`cls_labels`）和偏移量（`bbox_labels`）。
# 最后，我们根据类别和偏移量的预测和标注值计算损失函数。为了代码简洁，这里没有评价测试数据集。
# 

# In[18]:


num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')


# ## [**预测目标**]
# 
# 在预测阶段，我们希望能把图像里面所有我们感兴趣的目标检测出来。在下面，我们读取并调整测试图像的大小，然后将其转成卷积层需要的四维格式。
# 

# In[19]:


X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()


# 使用下面的`multibox_detection`函数，我们可以根据锚框及其预测偏移量得到预测边界框。然后，通过非极大值抑制来移除相似的预测边界框。
# 

# In[20]:


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)


# 最后，我们[**筛选所有置信度不低于0.9的边界框，做为最终输出**]。
# 

# In[21]:


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)


# ## 小结
# 
# * 单发多框检测是一种多尺度目标检测模型。基于基础网络块和各个多尺度特征块，单发多框检测生成不同数量和不同大小的锚框，并通过预测这些锚框的类别和偏移量检测不同大小的目标。
# * 在训练单发多框检测模型时，损失函数是根据锚框的类别和偏移量的预测及标注值计算得出的。
# 
# ## 练习
# 
# 1. 你能通过改进损失函数来改进单发多框检测吗？例如，将预测偏移量用到的$L_1$范数损失替换为平滑$L_1$范数损失。它在零点附近使用平方函数从而更加平滑，这是通过一个超参数$\sigma$来控制平滑区域的：
# 
# $$
# f(x) =
#     \begin{cases}
#     (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
#     |x|-0.5/\sigma^2,& \text{otherwise}
#     \end{cases}
# $$
# 
# 当$\sigma$非常大时，这种损失类似于$L_1$范数损失。当它的值较小时，损失函数较平滑。
# 

# In[22]:


def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();


# 此外，在类别预测时，实验中使用了交叉熵损失：设真实类别$j$的预测概率是$p_j$，交叉熵损失为$-\log p_j$。我们还可以使用焦点损失 :cite:`Lin.Goyal.Girshick.ea.2017`：给定超参数$\gamma > 0$和$\alpha > 0$，此损失的定义为：
# 
# $$ - \alpha (1-p_j)^{\gamma} \log p_j.$$
# 
# 可以看到，增大$\gamma$可以有效地减少正类预测概率较大时（例如$p_j > 0.5$）的相对损失，因此训练可以更集中在那些错误分类的困难示例上。
# 

# In[23]:


def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();


# 2. 由于篇幅限制，我们在本节中省略了单发多框检测模型的一些实现细节。你能否从以下几个方面进一步改进模型：
#     1. 当目标比图像小得多时，模型可以将输入图像调大。
#     1. 通常会存在大量的负锚框。为了使类别分布更加平衡，我们可以将负锚框的高和宽减半。
#     1. 在损失函数中，给类别损失和偏移损失设置不同比重的超参数。
#     1. 使用其他方法评估目标检测模型，例如单发多框检测论文 :cite:`Liu.Anguelov.Erhan.ea.2016`中的方法。
# 

# [Discussions](https://discuss.d2l.ai/t/3204)
# 
