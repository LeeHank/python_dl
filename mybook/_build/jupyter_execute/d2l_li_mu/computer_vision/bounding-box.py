#!/usr/bin/env python
# coding: utf-8

# # Bounding box

# ## import packages & utils

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import matplotlib.pyplot as plt
# from d2l import torch as d2l


# * 先讀一張圖：

# In[17]:


img = plt.imread('../img/catdog.jpg')
print(img.shape)
print(img.dtype)
print(img.max(), img.min())
plt.imshow(img);


# ## 兩種表法 & 畫圖

# * deep learning 的 alg 中，bounding box 表示方法是以下兩種：  
#   * corner: 用左上的 (xmin, ymin) 和 左下的 (xmax, ymax)，來表示 bounding box，所以是用 [xmin, ymin, xmax, ymax] 這種表法
#   * center & width/height: 用框框的中心 (x, y)，和框框的 width, height 來表示 bounding box，所以是用 [x, y, width, height] 這種方式
# * 特別注意：
#   * 之前都是把 "圖" 看成 2-d array，所以是用 (row, column) 的表法，例如 (2,3) 是指第 2 row 和 第 3 column 的像素
#   * 但這種表法，如果用直角坐標的角度來看， (row, column) 就是 (y, x) 的意思，不是 (x, y)
#   * 所以，現在在畫 bounding box 時，看到 (x, y) = (2, 3)，腦筋要轉一下，他不是指 2d-array 的 (row2, column3)，而是指直角座標的(x=2, y=3)，所以實際是(row3, col2)。
#   * 要特別小心這個 convention!! 做 bounding box 和 畫圖畫 bounding box 時，用的是 (x,y)表法(i.e. (col, row)表法)，但在做 影像處理/矩陣操作 時，用的是 (row, col) 表法 (i.e. (y, x)) 表法
# * 另外，在用 matplotlib 畫圖時，他吃的格式是 (xmin, ymin, width, height)，所以等等定義畫圖 function 時，還需要再轉一次

# ### corner 表法

# * 舉例來說，藉由人工標記，得到 dog 和 cat 的 bounding box 如下 (corner 表法)

# In[23]:


# bbox = bounding box
dog_bbox = [60.0, 45.0, 378.0, 516.0]
cat_bbox = [400.0, 112.0, 655.0, 493.0]


# * 我們可以定義一個畫圖 function 來看看效果：

# In[34]:


def bbox_to_rect(bbox, color, bbox_type = "corner"):
    '''
    args:
      - bbox: corner 表法： [xmin, ymin, xmax, ymax]
      - bbox_type: corner = corner 表法 (xmin, ymin, xmax, ymax)， center 表 center 表法 (cx, cy, width, height)
      - color: "blue", "red", ... 就 color
    remarks:
      - plt 的寫法，是要給他左上角的 (x1, y1)，和 width, height; 所以照他的格式畫圖
    '''
    
    if bbox_type == "corner":
        x = bbox[0]
        y = bbox[1]
        width = bbox[2]-bbox[0]
        height = bbox[3] - bbox[1]
    if bbox_type == "center":
        width = bbox[2]
        height = bbox[3]
        x = bbox[0] - 0.5*width
        y = bbox[1] - 0.5*height
        
    out = plt.Rectangle(
        xy=(x, y), # 左上角的 (x,y)
        width=width,
        height=height,
        fill=False, 
        edgecolor=color, 
        linewidth=2
    )
    return out


# In[26]:


fig = plt.imshow(img);
fig.axes.add_patch(bbox_to_rect(bbox = dog_bbox, color = 'blue', bbox_type = "corner"));
fig.axes.add_patch(bbox_to_rect(bbox = cat_bbox, color = 'red', bbox_type = "corner"));


# ### center + w/h 表法

# In[31]:


dog_bbox = [219, 280.5, 318, 471]
cat_bbox = [527.5, 302.5, 255, 381]


# In[35]:


fig = plt.imshow(img);
fig.axes.add_patch(bbox_to_rect(bbox = dog_bbox, color = 'blue', bbox_type = "center"));
fig.axes.add_patch(bbox_to_rect(bbox = cat_bbox, color = 'red', bbox_type = "center"));


# ## 表法轉換

# In[19]:


def box_corner_to_center(boxes):
    '''
    boxes 的 shape 為 (box個數, 4), 第二軸的 4 分別代表 corner 表法的 x1, x2, y1, y2
    '''
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    '''
    boxes 的 shape 為 (box個數, 4), 第二軸的 4 分別代表 center 表法的 x, y, width, height
    '''
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


# * 確認一下，轉回去再轉回來的值是 ok 的

# In[21]:


corner_boxes = torch.tensor((dog_bbox, cat_bbox))
center_boxes = box_corner_to_center(corner_boxes)
conrner_boxes_reconstruct = box_center_to_corner(center_boxes)
corner_boxes == conrner_boxes_reconstruct


# In[33]:


corner_boxes


# In[22]:


center_boxes

