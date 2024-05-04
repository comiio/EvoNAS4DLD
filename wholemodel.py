import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import Init_dld as init
import numpy as np

transform = transforms.Compose([
#     transforms.CenterCrop(224),

    transforms.RandomCrop(32,padding=4), # 数据增广
    transforms.RandomHorizontalFlip(),  # 数据增广
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

Batch_Size = 256

tr_img, tr_lab, val_img, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)

tr_img=torch.from_numpy(tr_img)
tr_lab=torch.from_numpy(tr_lab)
tst_img=torch.from_numpy(tst_img)
tst_lab=torch.from_numpy(tst_lab)

prob,tr_lab=tr_lab.data.max(dim=1)
prob,tst_lab=tst_lab.data.max(dim=1)

tr_img=tr_img.permute(0, 3, 1, 2)
tst_img=tst_img.permute(0, 3, 1, 2)
classes = {0: 'CON',1: 'MUL_GGO',2: 'HCM',3: 'RET_GGO',4: 'EMP',5: 'NOD',6: 'NOR'}
trainset=torch.utils.data.TensorDataset(tr_img,tr_lab)
testset=torch.utils.data.TensorDataset(tst_img,tst_lab)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size,shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size,shuffle=True, num_workers=8)


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

class BasicBlock(nn.Module):
    """
    对于浅层网络，如ResNet-18/34等，用基本的Block
    基础模块没有压缩,所以expansion=1
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
            
    def forward(self, x,switch=[]):
        halflen=int(len(switch)/2)
        out = self.conv1(x)
        for i in range(0,halflen):
            if(switch[i]==0):
                out[:,i,:,:]=float(0)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        for i in range(halflen,len(switch)):
            if(switch[i]==0):
                out[:,i-halflen,:,:]=0
        out = self.bn2(out)
        
        shortcut = self.shortcut(x)
        
        out += shortcut
        out = torch.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """
    对于深层网络，我们使用BottleNeck，论文中提出其拥有近似的计算复杂度，但能节省很多资源
    zip_channels: 压缩后的维数，最后输出的维数是 expansion * zip_channels
    针对ResNet50/101/152的网络结构,主要是因为第三层是第二层的4倍的关系所以expansion=4
    """
    expansion = 4
    
    def __init__(self, in_channels, zip_channels, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * zip_channels
        
        self.conv1 = nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(zip_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(zip_channels, zip_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(zip_channels)
        
        self.conv3 = nn.Conv2d(zip_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x,switch=[]):
        halflen=int(len(switch)/6)
        out = self.conv1(x)
        for i in range(0,halflen):
            if(switch[i]==0):
                out[:,i,:,:]=float(0)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        for i in range(halflen,halflen*2):
            if(switch[i]==0):
                out[:,i-halflen,:,:]=0
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        for i in range(halflen*2,len(switch)):
            if(switch[i]==0):
                out[:,i-halflen-halflen,:,:]=0
        out = self.bn3(out)
        
        shortcut = self.shortcut(x)
        
        out += shortcut
        out = torch.relu(out)
        
        return out


import torch.nn as nn

class ResNet(nn.Module):
    """
    不同的ResNet架构都是统一的一层特征提取、四层残差，不同点在于每层残差的深度。
    对于cifar10，feature map size的变化如下：
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2] 
    -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool] 
    -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """
    def __init__(self, block, num_blocks, num_classes=7, verbose=False):
        super(ResNet, self).__init__()
        self.verbose = verbose
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out1=64
        self.out2=128
        self.out3=256
        self.out4=512
        # 使用_make_layer函数生成上表对应的conv2_x, conv3_x, conv4_x, conv5_x的结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # cifar10经过上述结构后，到这里的feature map size是 4 x 4 x 512 x expansion
        # 所以这里用了 4 x 4 的平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifer = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block要进行降采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layer = block(self.in_channels, out_channels, stride)
            layers.append(layer)
            # 如果是Bottleneck Block的话需要对每层输入的维度进行压缩，压缩后再增加维数
            # 所以每层的输入维数也要跟着变
            self.in_channels = out_channels * block.expansion
        return layers
    
    def forward(self, x, switch=[1] * 3776*2):
        out = self.features(x)
        if self.verbose:
            print('block 1 output: {}'.format(out.shape))
        
        i=0

        for layer in self.layer1:
                out = layer(out,switch[i:i+self.out1*2])
                i = i+self.out1*2
        if self.verbose:
            print('block 2 output: {}'.format(out.shape))

        for layer in self.layer2:
            out = layer(out,switch[i:i+self.out2*2])
            i = i+self.out2*2
        if self.verbose:
            print('block 3 output: {}'.format(out.shape))

        for layer in self.layer3:
            out = layer(out,switch[i:i+self.out3*2])
            i = i+self.out3*2
        if self.verbose:
            print('block 4 output: {}'.format(out.shape))
            
        for layer in self.layer4:
            out = layer(out,switch[i:i+self.out4*2])
            i = i+self.out4*2
        
        if self.verbose:
            print('block 5 output: {}'.format(out.shape))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        i=0
        return out

class ResNet2(nn.Module):
    """
    不同的ResNet架构都是统一的一层特征提取、四层残差，不同点在于每层残差的深度。
    对于cifar10，feature map size的变化如下：
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2] 
    -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool] 
    -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """
    def __init__(self, block, num_blocks, num_classes=7, verbose=False):
        super(ResNet2, self).__init__()
        self.verbose = verbose
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out1=64
        self.out2=128
        self.out3=256
        self.out4=512
        # 使用_make_layer函数生成上表对应的conv2_x, conv3_x, conv4_x, conv5_x的结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # cifar10经过上述结构后，到这里的feature map size是 4 x 4 x 512 x expansion
        # 所以这里用了 4 x 4 的平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifer = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block要进行降采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layer = block(self.in_channels, out_channels, stride)
            layers.append(layer)
            # 如果是Bottleneck Block的话需要对每层输入的维度进行压缩，压缩后再增加维数
            # 所以每层的输入维数也要跟着变
            self.in_channels = out_channels * block.expansion
        return layers
    
    def forward(self, x, switch=[1] * 22656):
        out = self.features(x)
        if self.verbose:
            print('block 1 output: {}'.format(out.shape))
        
        i=0
        for layer in self.layer1:
                out = layer(out,switch[i:i+self.out1*6])
                i = i+self.out1*6
        if self.verbose:
            print('block 2 output: {}'.format(out.shape))

        for layer in self.layer2:
            out = layer(out,switch[i:i+self.out2*6])
            i = i+self.out2*6
        if self.verbose:
            print('block 3 output: {}'.format(out.shape))

        for layer in self.layer3:
            out = layer(out,switch[i:i+self.out3*6])
            i = i+self.out3*6
        if self.verbose:
            print('block 4 output: {}'.format(out.shape))
            
        for layer in self.layer4:
            out = layer(out,switch[i:i+self.out4*6])
            i = i+self.out4*6
        if self.verbose:
            print('block 5 output: {}'.format(out.shape))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        i=0
        return out

def ResNet18(verbose=False):
    return ResNet(BasicBlock, [2,2,2,2],verbose=verbose)

def ResNet34(verbose=False):
    return ResNet(BasicBlock, [3,4,6,3],verbose=verbose)

def ResNet50(verbose=False):
    return ResNet2(Bottleneck, [3,4,6,3],verbose=verbose)

def ResNet101(verbose=False):
    return ResNet2(Bottleneck, [3,4,23,3],verbose=verbose)

def ResNet152(verbose=False):
    return ResNet2(Bottleneck, [3,8,36,3],verbose=verbose)

#net = ResNet18(True).to(device)

net = ResNet34().to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    torch.backends.cudnn.benchmark = True

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,verbose=True,patience = 5,min_lr = 0.000001) # 动态更新学习率
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

import time
epoch = 100

import os
if not os.path.exists('./model'):
    os.makedirs('./model')
else:
    print('文件已存在')
save_path = './model/ResNet34.pth'

from utils import train
from utils import plot_history
Acc, Loss, Lr = train(net, trainloader, testloader, epoch, optimizer, criterion, scheduler, save_path, verbose = True)

def fitness(ori_acc):
    def eval_func(state_switch):
        #new_acc,_ = validation_procedure(val_img=val_img,val_lab=val_lab,switch=state_switch)
        new_acc = float(test(switch=state_switch))
        #print(new_acc)
        '''if caonima > 0.9478:
            drop_num= ' '.join([str(i) for i in state_switch])
            save2file(drop_num)'''
        return new_acc/ori_acc
    return eval_func

def test(switch=[]):
    correct = 0   # 定义预测正确的图片数，初始化为0
    total = 0     # 总共参与测试的图片数，也初始化为0
    torch.cuda.empty_cache()
    net.eval()
    with torch.no_grad():
        for data in testloader:  # 循环每一个batch
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            net.eval()  # 把模型转为test模式
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = net(images,switch)  # 输入网络进行测试

        # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)          # 更新测试图片的数量
            correct += (predicted == labels).sum() # 更新正确分类的图片的数量
    return correct/total

#剪枝
import ESEA
Pruning=True
if Pruning:
        filter_drop = []
        print('Pruning procedure :')
        p_size=[1] * 3776*2
        ori_acc= float(test(switch=p_size))
        st_time = time.time()
        ea_helper = ESEA.EA_Util(
            'es_ea', 
            30, 
            3776*2, 
            eval_fun=fitness(ori_acc), 
            max_gen=50, 
            drop_set=filter_drop, 
            sp_set=[],
            target='F'
        )
        elite = ea_helper.evolution()
        elite = ea_helper.population[elite]
        en_time = time.time()
        for x in range(3776*2):
            if elite[x] == 0:
                filter_drop.append(x)
        print('Drop Set:', filter_drop)
        print('Drop Count:', len(filter_drop))
        print('GA fliter Consume:', int((en_time-st_time)*1000))