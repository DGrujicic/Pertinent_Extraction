
# coding: utf-8

# In[ ]:


from network2 import load_network2, write_positives_network2, Net_Big_Image, Block_First, Block_Last, Block
from utilities import get_transform
import torch
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
import os.path

batch_size = 64
img_size = 256
threshold = 0.65

data_transform =  get_transform(img_size)
filename = 'model2_positives_2018-03-20_v3.txt'
data_dir = '/esat/nihal/arannen/Lung_data/Emphysema_png_Rearranged'
use_gpu = torch.cuda.is_available()
use_gpu = False
print("GPU available: {}".format(use_gpu))
#model = load_network2(state_dict = 'pretrained_state_dict.pth')
model = load_network2(state_dict = 'model2_cpu_state_dict_2018-03-18.pth')
model.eval()
#model = torch.load('model_small_cpu_91acc_91rec.pth')
print(model)
write_positives_network2(model,data_dir,data_transform,batch_size,threshold,filename,use_gpu)

