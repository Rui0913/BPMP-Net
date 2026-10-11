import cv2
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from os.path import join
import cv2



transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_pan = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5], std=[0.5])
])


class FlameSet():
       def __init__(self,image_dir):

              input_pan_dir = join(image_dir, "pan")
              pan = os.listdir(input_pan_dir)
              self.pan = [os.path.join(input_pan_dir, k) for k in pan]
              input_ms_dir = join(image_dir, "ms")
              ms = os.listdir(input_ms_dir)
              self.ms = [os.path.join(input_ms_dir, k) for k in ms]


              gt_dir = join(image_dir, "label")
              gt = os.listdir(gt_dir)
              self.gt = [os.path.join(gt_dir, k) for k in gt]
              self.filename = [os.path.join(gt_dir, k).split('/')[-1] for k in gt]

              self.pan_transform = transform_pan
              self.ms_transform = transform
              self.target_transform = transform

       def __getitem__(self, index):
              pan_path = self.pan[index]
              pan_img = cv2.imread(pan_path, flags = 0)
              pan_img = np.reshape(pan_img, (256, 256, 1))

              ms_path = self.ms[index]
              ms_img = cv2.imread(ms_path, flags = 1)
              ms_img = np.reshape(ms_img, (64, 64, 3))

              gt_path = self.gt[index]
              gt_img = cv2.imread(gt_path, flags = 1)
              gt_img = np.reshape(gt_img, (256, 256, 3))

              gt_filename = self.filename[index]

              if self.pan_transform:
                     input_pan = self.pan_transform(pan_img)

              if self.ms_transform:
                     input_ms = self.ms_transform(ms_img)

              if self.target_transform:
                     target = self.target_transform(gt_img)

              return input_pan, input_ms, target, gt_filename

       def __len__(self):
              return len(self.pan)

class DataPrefetcher():     #在 GPU 训练模型时，提前加载数据，以减少训练过程中的数据加载延迟
       def __init__(self, loader):
              self.loader = iter(loader)    #将 loader 转换为一个迭代器（使用 iter），便于后续逐批次读取数据
              self.stream = torch.cuda.Stream()    #创建一个新的 CUDA 流。CUDA 流用于在 GPU 上执行操作，允许异步执行，以提高性能。
              self.preload()

       def preload(self):
              try:
                     self.batch = next(self.loader)   #尝试获取下一个批次的数据，并将其存储在 self.batch 中。如果获取成功，self.batch 将包含当前批次的数据
              except StopIteration:
                     self.batch = None  #处理迭代器结束的情况（即没有更多的数据可供加载）
                     return

       def next(self):   #用于获取当前批次的数据
              torch.cuda.current_stream().wait_stream(self.stream)  #等待当前 CUDA 流完成执行。这是为了确保在使用 self.batch 时，先前的 GPU 操作已经完成，避免潜在的数据竞态条件。
              batch = self.batch  #将当前的批次数据赋值给 batch 变量
              self.preload()
              return batch