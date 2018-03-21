
# coding: utf-8

# In[1]:


from network2 import load_network2, train_network2
from optimization import StepLR, BCEWithLogitsLoss
from utilities import get_transforms
import torch
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
import os.path

batch_size = 64
num_epochs = 20
img_size = 256
threshold = 0.6

# Optimization hyperparameters
learning_rate = 0.0001 #0.05 as a starting point works better for fine tuning
momentum = 0.9
step_size = 5
gamma = 0.5 # multiply the learning rate by this every step_size epochs


model_name = '-----------Network_Small_1fc-----------'
param_list = ['Learning rate {}'.format(learning_rate), 'Momentum {}'.format(momentum), 'Step size {}'.format(step_size), 'Gamma {}'.format(gamma), 'Optimizer Adam']
data_transforms = get_transforms(img_size)

data_dir = '/esat/nihal/arannen/Lung_data/Emphysema_png_Rearranged'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size, num_workers=8, shuffle=True) for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print("Dataset size:{}".format(dataset_sizes))
use_gpu = torch.cuda.is_available()
print("GPU available: {}".format(use_gpu))
model = load_network2(state_dict = 'model_cpu_net2_state_dict.pth')
print(model)

optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0)
exp_lr_scheduler = StepLR(optimizer, step_size = step_size, gamma = gamma)
criterion = BCEWithLogitsLoss()

model = train_network2(model, model_name, param_list, dataloaders, criterion, optimizer, exp_lr_scheduler, dataset_sizes, threshold, use_gpu, num_epochs)


# In[2]:




