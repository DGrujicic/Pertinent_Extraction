import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from utilities import ImageFolderCustom
import numpy as np

def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1):    
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)

class Block_First(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(Block_First, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = ConvBlock(self.in_channels, self.out_channels, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = ConvBlock(self.out_channels, self.out_channels, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.act2 = nn.ReLU()
        
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = F.relu(x)
        if self.pooling:
            x = self.pool(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(Block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = ConvBlock(self.in_channels, self.out_channels, 3, 1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = ConvBlock(self.out_channels, self.out_channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.act2 = nn.ReLU()
        
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = F.relu(x)
        if self.pooling:
            x = self.pool(x)
        return x
    
class Block_Last(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(Block_Last, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = ConvBlock(self.in_channels, self.out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = ConvBlock(self.out_channels, self.out_channels, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.act2 = nn.ReLU()
        
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = F.relu(x)
        if self.pooling:
            x = self.pool(x)
        return x
    
class Net_Big_Image(nn.Module):
    
    def __init__(self, num_classes, in_channels=3, depth=5, num_fc = 2, 
                 out_channels_first_block=64, img_size = 256):

        super(Net_Big_Image, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels_first_block = out_channels_first_block
        self.depth = depth
        self.num_fc = num_fc
        self.img_size = img_size

        # Setting Up Blocks of Two Convolutions and One Max Pooling
        self.blocks = []
        self.fully_connecteds = []
        
        # First Block
        block = Block_First(self.in_channels, out_channels_first_block, pooling=True)
        out_channels_block = self.out_channels_first_block
        self.blocks.append(block)
        
        # Other Blocks
        for i in range(depth-1):
            in_channels_block = out_channels_block
            out_channels_block = self.out_channels_first_block*(2**(i+1))
            if not i==depth-2:
                block = Block(in_channels_block, out_channels_block, pooling=True)
            else:
                block = Block_Last(in_channels_block, int(out_channels_block/2), pooling=True)
            self.blocks.append(block)
            
        # Last Block

        self.blocks = nn.ModuleList(self.blocks)
        
        # Setting Up the Fully Connected Layers
        out_channels =self.blocks[self.depth-1].conv2.out_channels
        out_img_size = self.img_size/(4*2**(self.depth)) # Would normally be self.depth-1 but poolings done after the last conv too
        self.in_num_first_fc = int(out_img_size**2 * out_channels)
        
        for i in range(num_fc):
            in_num_fc = self.in_num_first_fc if i == 0 else out_num_fc
            out_num_fc = self.num_classes if i == num_fc-1 else int(in_num_fc/16)
            fully_connected=nn.Linear(in_num_fc, out_num_fc)
            self.fully_connecteds.append(fully_connected)
            if i<self.num_fc-1:
                bn = nn.BatchNorm1d(out_num_fc)
                self.fully_connecteds.append(bn)
                act = nn.ReLU()
                self.fully_connecteds.append(act)
        
        self.fully_connecteds = nn.ModuleList(self.fully_connecteds)
        self.initialize()

    @staticmethod
    def weight_init(module): # kaiming and xavier seem to perform the best
        if isinstance(module, nn.Conv2d):
            init.xavier_normal(module.weight)
            # init.constant(module.bias, 0)
        if isinstance(module, nn.Linear):
            init.xavier_normal(module.weight)
            init.constant(module.bias, 0)

    def initialize(self):
        for i, module in enumerate(self.modules()):
            self.weight_init(module)

    def forward(self, x):
        # Pass Through Blocks
        for i, module in enumerate(self.blocks):
            x = module(x)  
        # Pass Through The Fully Connected Layer
        x = x.view(-1, self.in_num_first_fc)
        for i, module in enumerate(self.fully_connecteds):
            x = module(x)
        return x
    
def write_positives_network2(model,data_dir,data_transform,batch_size,threshold,filename,use_gpu):
    file = open(filename, "w")
    dataset = ImageFolderCustom(data_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, shuffle=False)
    if use_gpu:
        model = model.cuda()
    model.train(False)
    for data in dataloader:
        inputs, paths, indices = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs)>threshold).squeeze(1).type(torch.LongTensor)

        for j in range(inputs.size()[0]):
            if np.asscalar(preds.data.numpy()[j])==0:
                file.write('\n {}'.format(paths[j]))

    file.close()
    
def load_network2(state_dict = 'network2_cpu_state_dict.pth'):
    model = Net_Big_Image(num_classes=1,in_channels=1,depth=5,num_fc=1,out_channels_first_block=16, img_size = 256)
    model.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage))
    return model

    
def train_network2(model, model_name, param_list, dataloaders, criterion, optimizer, scheduler, dataset_sizes, threshold=0.5, use_gpu=False, num_epochs=25):
    import datetime
    import time
    now = datetime.datetime.now()
    file_log = open('model2_train_log_{}.txt'.format(str(now.date())), 'w')
    file_log.write("%s\n" % model_name)
    for item in param_list:
        file_log.write("%s\n" % item)
    train_log = []
    since = time.time()
    """i=0"""
    if use_gpu:
        model = model.cuda()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True) # Set model to traning mode
            else:
                model.train(False) # Set model to evaluate mode
                   
            running_loss = 0.0
            running_corrects = 0
            # For Confusion Matrix
            running_corrects_useless = 0
            running_corrects_useful = 0
            running_total_useless = 0
            running_total_useful = 0
            
            for data in dataloaders[phase]:
                inputs, labels = data
                
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    labels_crit = labels.unsqueeze(1).type(torch.FloatTensor).cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    labels_crit = labels.unsqueeze(1).type(torch.FloatTensor)
                    
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs)>threshold).squeeze(1).type(torch.LongTensor)
                if use_gpu:
                    preds = preds.cuda()
                loss = criterion(outputs, labels_crit)
                

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.data == labels.data)
                # For Confusion Matrix
                running_corrects_useless += torch.sum((preds.data+labels.data) == 2)
                running_corrects_useful += torch.sum((preds.data+labels.data) == 0)
                running_total_useless += torch.sum(labels.data == 1)
                running_total_useful += torch.sum(labels.data == 0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            train_log.append('Epoch: {}/{} Phase: {} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f}'.format(epoch, num_epochs - 1, phase, epoch_loss, epoch_acc, running_corrects_useful/(running_corrects_useful+running_total_useless-running_corrects_useless), running_corrects_useful/running_total_useful))
            print('{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f}'.format(phase, epoch_loss, epoch_acc, running_corrects_useful/(running_corrects_useful+running_total_useless-running_corrects_useless), running_corrects_useful/running_total_useful))
            # deep copy the model           
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'model2_state_dict_checkpoint_{}.pth'.format(str(now.date()))) #Check if it's actually in cpu readable format
 
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    for item in train_log:
        file_log.write("%s\n" % item)
    file_log.close()
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    if use_gpu:
        model = model.cpu()
    torch.save(model, './model2_{}.pth'.format(str(now.date())))
    torch.save(model.state_dict(), 'model2_state_dict_final_{}.pth'.format(str(now.date())))
    return model

