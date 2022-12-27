#!/usr/bin/env python
# coding: utf-8

# # Formula Trinity 🤝 DUCSS - Object Detection Workshop!
# 
# Hello Everyone!
# 
# Today we will be going through how to make a traffic cone detector using [PyTorch](https://pytorch.org/).
# 
# This notebook was adapted from the one in [Fine-tuning Faster-RCNN using pytorch](https://www.kaggle.com/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook)
# 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
os.chdir("/content/drive/MyDrive/0. codepool_python/python_dl/mybook/others")


# ## Installs and Imports
# 
# Install and import PyTorch along with a few helper libraries

# Let's install some dependencies and clone the [TorchVision Repo](https://github.com/pytorch/vision) so we can use some helper files

# In[ ]:


# Install dependencies and 
get_ipython().system('pip install albumentations==0.4.6')
get_ipython().system('pip install pycocotools --quiet')

# Clone TorchVision repo and copy helper files
get_ipython().system('git clone https://github.com/pytorch/vision.git')
get_ipython().run_line_magic('cd', 'vision')
get_ipython().system('git checkout v0.3.0')
get_ipython().run_line_magic('cd', '..')
get_ipython().system('cp vision/references/detection/utils.py ./')
get_ipython().system('cp vision/references/detection/transforms.py ./')
get_ipython().system('cp vision/references/detection/coco_eval.py ./')
get_ipython().system('cp vision/references/detection/engine.py ./')
get_ipython().system('cp vision/references/detection/coco_utils.py ./')


# In[ ]:


# basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# helper libraries
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# Lets import the libraries

# ## The Dataset 

# Having a good dataset is essential to training an accurate model. However, just possessing the dataset is not enough. We must pre-process it so that it can be fed to the neural network - and then understood by it.
# 
# Finding an existing dataset can sometimes be very difficult. When searching for your dataset you may run into the following problems:
# 
# * **Broken links**: even if someone has open-sourced something, links they previously had to their datasets often end up becoming stale.
# * **Poor Open-Source Practice**: some "open-source" dataset providers will insist on you contributing to the dataset to be provided access.
# 
# However, keep searching! Here's some tricks:
# 
# * **Search specific websites**: When searching on Google, you can search specifically on GitHub by searching something like: `cone detection dataset site:github.com`.
# * **Reading other people's code**: Try to find the code of someone else that worked on a similar problem, and see if you can find out what dataset they used! Such as searching: `image classification on traffic cones` and trying to find other notebooks.
# 
# 
# We found the dataset for this worksop by scouring GitHub repos. It contains 123 annotated images in the training set and 
# 
# Let's download it!

# In[ ]:


get_ipython().system('wget -O data.rar https://www.dropbox.com/sh/ay9wf7ii81q5zif/AADwIb9HkvpBmUDJvKpNl0Xna?dl=0&file_subpath=%2Fimg&preview=data.rar#:~:text=Sign%20up-,Direct,-download')
get_ipython().system('unrar x data.rar')
# image 96 had no annotations, so we'll delete it!
get_ipython().system('rm img/96.jpg')


# In[ ]:


# defining the files directory and testing directory
files_dir = '/content/drive/MyDrive/0. codepool_python/python_dl/mybook/others/img/'
test_dir = '/content/drive/MyDrive/0. codepool_python/python_dl/mybook/others/img1/'

# we create a Dataset class which has a __getitem__ function and a __len__ function
class ConeImagesDataset(torch.utils.data.Dataset):

  def __init__(self, files_dir, width, height, transforms=None):
    self.transforms = transforms
    self.files_dir = files_dir
    self.height = height
    self.width = width
    
    # sorting the images for consistency
    # To get images, the extension of the filename is checked to be jpg
    self.imgs = [image for image in sorted(os.listdir(files_dir)) if image[-4:]=='.jpg']
    
    # classes: 0 index is reserved for background
    self.classes = [_, 'cone']

  def __getitem__(self, idx):
    img_name = self.imgs[idx]
    image_path = os.path.join(self.files_dir, img_name)

    # reading the images and converting them to correct size and color    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
    # diving by 255
    img_res /= 255.0
    
    # annotation file
    annot_filename = img_name[:-4] + '.txt'
    annot_file_path = os.path.join(self.files_dir, annot_filename)
    
    boxes = []
    labels = []
    
    # cv2 image gives size as height x width
    wt = img.shape[1]
    ht = img.shape[0]
    
    # box coordinates for xml files are extracted and corrected for image size given
    with open(annot_file_path) as f:
      for line in f:
        labels.append(1)
        
        parsed = [float(x) for x in line.split(' ')]
        x_center = parsed[1]
        y_center = parsed[2]
        box_wt = parsed[3]
        box_ht = parsed[4]

        xmin = x_center - box_wt/2
        xmax = x_center + box_wt/2
        ymin = y_center - box_ht/2
        ymax = y_center + box_ht/2
        
        xmin_corr = int(xmin*self.width)
        xmax_corr = int(xmax*self.width)
        ymin_corr = int(ymin*self.height)
        ymax_corr = int(ymax*self.height)
        
        boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
    
    # convert boxes into a torch.Tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
    # getting the areas of the boxes
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # suppose all instances are not crowd
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    
    labels = torch.as_tensor(labels, dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["area"] = area
    target["iscrowd"] = iscrowd
    image_id = torch.tensor([idx])
    target["image_id"] = image_id

    if self.transforms:
      sample = self.transforms(image = img_res,
                                bboxes = target['boxes'],
                                labels = labels)
      img_res = sample['image']
      target['boxes'] = torch.Tensor(sample['bboxes'])
        
    return img_res, target

  def __len__(self):
    return len(self.imgs)


# check dataset
dataset = ConeImagesDataset(files_dir, 224, 224)
print('Length of dataset:', len(dataset), '\n')

# getting the image and target for a test index.  Feel free to change the index.
img, target = dataset[78]
print('Image shape:', img.shape)
print('Label example:', target)


# # Visualization
# 
# Let's make some a helper function to view our data

# In[ ]:


# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
  # plot the image and bboxes
  # Bounding boxes are defined as follows: x-min y-min width height
  fig, a = plt.subplots(1,1)
  fig.set_size_inches(5,5)
  a.imshow(img)
  for box in (target['boxes']):
    x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
    rect = patches.Rectangle(
      (x, y),
      width, height,
      linewidth = 2,
      edgecolor = 'r',
      facecolor = 'none'
    )
    # Draw the bounding box on top of the image
    a.add_patch(rect)
  plt.show()
    
# plotting the image with bboxes. Feel free to change the index
img, target = dataset[25]
plot_img_bbox(img, target)


# In[ ]:


target


# # Augmentations

# This is where we can apply augmentations to the image. 
# 
# The augmentations to object detection vary from normal augmentations becuase here we need to ensure that, bbox still aligns with the object correctly after transforming.
# 
# Here we are doing a random flip transform.
# 
# 

# In[ ]:


# Send train=True for training transforms and False for val/test transforms
def get_transform(train):
  if train:
    return A.Compose(
      [
        A.HorizontalFlip(0.5),
        # ToTensorV2 converts image to pytorch tensor without div by 255
        ToTensorV2(p=1.0) 
      ],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return A.Compose(
      [ToTensorV2(p=1.0)],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


# # Dataloaders
# 
# Make a loader for feeding our data into the neural network

# Now lets prepare the datasets and dataloaders for training and testing.

# In[ ]:


# use our dataset and defined transformations
dataset = ConeImagesDataset(files_dir, 480, 480, transforms=get_transform(train=True))
dataset_test = ConeImagesDataset(files_dir, 480, 480, transforms=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
  dataset,
  batch_size=10,
  shuffle=True,
  num_workers=4,
  collate_fn=utils.collate_fn,
)

data_loader_test = torch.utils.data.DataLoader(
  dataset_test,
  batch_size=10,
  shuffle=False,
  num_workers=4,
  collate_fn=utils.collate_fn,
)


# # Pre-trained Model

# In[ ]:


def get_object_detection_model(num_classes):
  # load a model pre-trained pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
  return model


# # Training

# Let's prepare the model for training

# In[ ]:


# train on gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2 # one class (class 0) is dedicated to the "background"

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
  optimizer,
  step_size=3,
  gamma=0.1
)


# Let the training begin!

# In[ ]:


# training for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)


# # Filtering the outputs

# Our model predicts a lot of bounding boxes per image, so take out the overlapping ones, we will use **Non Max Suppression** (NMS). If you want to brush up on that, check [this](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) out.
# 
# Torchvision provides us a utility to apply NMS to our predictions, lets build a function `apply_nms` using that.

# In[ ]:


# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep]
  final_prediction['scores'] = final_prediction['scores'][keep]
  final_prediction['labels'] = final_prediction['labels'][keep]
  
  return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
  return torchtrans.ToPILImage()(img).convert('RGB')


# # Testing our Model

# Now lets take an image from the test set and try to predict on it

# In[ ]:


test_dataset = ConeImagesDataset(test_dir, 480, 480, transforms= get_transform(train=True))

# pick one image from the test set
img, target = test_dataset[10]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
  prediction = model([img.to(device)])[0]
    
print('MODEL OUTPUT\n')
nms_prediction = apply_nms(prediction, iou_thresh=0.01)
nms_prediction2 = {key: value.cpu() for key, value in nms_prediction.items()}

plot_img_bbox(torch_to_pil(img), nms_prediction2)

