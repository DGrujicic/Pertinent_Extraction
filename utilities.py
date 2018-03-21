
# coding: utf-8

# In[3]:


import torch
from torch.autograd import Variable
import os
import os.path
from PIL import Image
import torch.utils.data as data
import scipy.ndimage as ndimage
import cv2
import numpy as np
import torchvision
from torchvision import datasets, transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

class Grayscale(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
    def __call__(self,img):
        return transforms.ImageOps.grayscale(img)
    
class Mask(object):
    def __call__(self,img):
        img = np.array(img)
        mask=(img>0).astype(bool)
        struct = ndimage.generate_binary_structure(2,2)
        mask = ndimage.binary_closing(mask,structure=struct)
        objects = ndimage.find_objects(mask)
        label_im, nb_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        mask_size = sizes < sizes.max()
        remove_pixel = mask_size[label_im]
        remove_pixel.shape
        label_im[remove_pixel] = 0
        mask = ndimage.binary_fill_holes(label_im)
        mask = mask.astype(np.uint8)
        img=cv2.bitwise_and(img,img,mask=mask)
        return Image.fromarray(img)

def pil_loader(path):
    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

import os
    
def get_all_images(dir):
    
    images = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames: #[f for f in filenames if f.endswith(tuple(IMG_EXTENSIONS))]:
            images.append(os.path.join(dirpath, filename))
    return images

class ImageFolderCustom(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_loader):
        imgs = get_all_images(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, path, index
    
    def __len__(self):
        return len(self.imgs)

def get_transforms(img_size):
    
    data_transforms = {
        'train': transforms.Compose([
            Grayscale(),
            Mask(),
            transforms.Scale(img_size),
            transforms.RandomSizedCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.163],[0.168])
        ]),
        'val': transforms.Compose([
            Grayscale(),
            Mask(),
            transforms.Scale(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.163],[0.168])
        ])
    }
    return data_transforms

def get_transform(img_size):
    
    data_transform =  transforms.Compose([
            Grayscale(),
            Mask(),
            transforms.Scale(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.163],[0.168])     
        ])
    return data_transform
