import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import Init_dld as init
import numpy as np
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

import torch.nn as nn

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

import torch

model=torch.load("model/ResNet50.pth")

model=model.module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


model.to(device)

import Init_dld as init

tr_img, tr_lab, val_img, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)


tr_img=torch.from_numpy(tr_img)
tr_lab=torch.from_numpy(tr_lab)
tst_img=torch.from_numpy(tst_img)
tst_lab=torch.from_numpy(tst_lab)

tr_img=tr_img.permute(0, 3, 1, 2)
tst_img=tst_img.permute(0, 3, 1, 2)

prob,tr_lab=tr_lab.data.max(dim=1)
prob,tst_lab=tst_lab.data.max(dim=1)
Batch_Size=128
trainset=torch.utils.data.TensorDataset(tr_img,tr_lab)
testset=torch.utils.data.TensorDataset(tst_img,tst_lab)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size,shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size,shuffle=False, num_workers=8)

drop=[15, 16, 33, 48, 53, 68, 70, 87, 93, 126, 155, 161, 163, 170, 191, 198, 230, 239, 328, 336, 353, 400, 402, 423, 431, 471, 492, 494, 502, 522, 557, 567, 589, 596, 605, 606, 618, 622, 625, 628, 631, 661, 675, 688, 746, 768, 777, 780, 797, 808, 845, 881, 896, 908, 909, 919, 920, 929, 931, 935, 951, 986, 1034, 1047, 1049, 1058, 1079, 1134, 1150, 1222, 1285, 1299, 1328, 1334, 1375, 1384, 1394, 1406, 1420, 1444, 1461, 1478, 1481, 1498, 1579, 1591, 1614, 1620, 1628, 1690, 1733, 1812, 1814, 1825, 1836, 1852, 1858, 1871, 1875, 1881, 1905, 1916, 1929, 1978, 2008, 2036, 2037, 2053, 2056, 2066, 2068, 2151, 2163, 2180, 2184, 2196, 2202, 2213, 2225, 2240, 2261, 2264, 2282, 2288, 2300, 2303, 2304, 2306, 2313, 2322, 2327, 2340, 2345, 2347, 2394, 2409, 2437, 2464, 2496, 2509, 2522, 2530, 2544, 2579, 2587, 2620, 2626, 2632, 2668, 2670, 2679, 2680, 2689, 2695, 2715, 2720, 2722, 2724, 2727, 2745, 2775, 2795, 2797, 2804, 2845, 2855, 2856, 2873, 2907, 2908, 2942, 2948, 2960, 2989, 3018, 3035, 3052, 3059, 3084, 3126, 3150, 3153, 3156, 3161, 3167, 3172, 3194, 3198, 3220, 3250, 3290, 3312, 3324, 3340, 3369, 3375, 3381, 3431, 3437, 3492, 3493, 3521, 3556, 3563, 3571, 3572, 3577, 3579, 3589, 3639, 3649, 3664, 3676, 3702, 3709, 3710, 3728, 3759, 3774, 3780, 3782, 3787, 3789, 3791, 3830, 3849, 3856, 3880, 3899, 3900, 3906, 3912, 3934, 3942, 3947, 3953, 3957, 3972, 3981, 3993, 4006, 4022, 4039, 4103, 4119, 4122, 4130, 4161, 4178, 4187, 4192, 4195, 4205, 4208, 4209, 4222, 4224, 4268, 4272, 4299, 4327, 4330, 4374, 4379, 4386, 4402, 4442, 4457, 4462, 4464, 4487, 4491, 4493, 4497, 4521, 4523, 4524, 4529, 4538, 4559, 4606, 4607, 4622, 4632, 4643, 4660, 4662, 4681, 4683, 4687, 4727, 4764, 4768, 4806, 4807, 4812, 4815, 4823, 4836, 4845, 4856, 4870, 4896, 4925, 4974, 4977, 4989, 4994, 5037, 5054, 5069, 5085, 5088, 5097, 5099, 5114, 5144, 5160, 5174, 5197, 5203, 5237, 5240, 5251, 5258, 5281, 5337, 5375, 5413, 5416, 5444, 5466, 5482, 5485, 5487, 5493, 5512, 5517, 5542, 5564, 5634, 5682, 5705, 5754, 5756, 5761, 5792, 5807, 5816, 5845, 5880, 5911, 5912, 5922, 5941, 5942, 5944, 5956, 6004, 6027, 6034, 6038, 6048, 6054, 6076, 6083, 6087, 6149, 6156, 6161, 6176, 6182, 6225, 6234, 6262, 6281, 6303, 6313, 6360, 6371, 6384, 6389, 6408, 6428, 6445, 6472, 6474, 6490, 6494, 6561, 6599, 6623, 6631, 6640, 6644, 6649, 6664, 6669, 6690, 6709, 6732, 6736, 6741, 6767, 6778, 6794, 6804, 6821, 6834, 6843, 6850, 6865, 6866, 6869, 6884, 6902, 6915, 6923, 6939, 6969, 6971, 6980, 6996, 7074, 7075, 7093, 7136, 7138, 7141, 7151, 7165, 7183, 7194, 7197, 7199, 7205, 7222, 7236, 7247, 7274, 7278, 7294, 7302, 7313, 7325, 7326, 7336, 7340, 7344, 7368, 7376, 7401, 7432, 7446, 7459, 7482, 7484, 7493, 7503, 7558, 7575, 7603, 7612, 7622, 7631, 7641, 7651, 7671, 7677, 7692, 7719, 7729, 7734, 7738, 7747, 7750, 7804, 7806, 7815, 7818, 7828, 7836, 7850, 7895, 7916, 7929, 7959, 7971, 7974, 7988, 7991, 7997, 8053, 8058, 8065, 8075, 8078, 8080, 8088, 8097, 8127, 8130, 8147, 8157, 8167, 8186, 8229, 8274, 8275, 8295, 8301, 8321, 8323, 8333, 8337, 8372, 8375, 8389, 8427, 8433, 8437, 8449, 8450, 8454, 8467, 8477, 8513, 8531, 8532, 8536, 8544, 8567, 8577, 8594, 8628, 8642, 8648, 8652, 8659, 8662, 8682, 8683, 8686, 8688, 8713, 8747, 8751, 8759, 8791, 8798, 8841, 8859, 8867, 8870, 8877, 8880, 8924, 8960, 8966, 8995, 9005, 9017, 9031, 9110, 9117, 9118, 9124, 9133, 9139, 9140, 9150, 9165, 9168, 9173, 9192, 9235, 9269, 9271, 9280, 9287, 9310, 9340, 9379, 9394, 9401, 9411, 9419, 9429, 9478, 9480, 9494, 9519, 9553, 9563, 9574, 9594, 9609, 9644, 9661, 9701, 9718, 9745, 9755, 9760, 9771, 9778, 9783, 9842, 9859, 9873, 9875, 9916, 9948, 9961, 10011, 10017, 10060, 10094, 10110, 10124, 10129, 10134, 10164, 10193, 10195, 10220, 10233, 10242, 10255, 10257, 10303, 10348, 10362, 10368, 10387, 10393, 10409, 10454, 10481, 10491, 10536, 10570, 10580, 10641, 10655, 10679, 10711, 10728, 10744, 10777, 10825, 10838, 10883, 10897, 10898, 10912, 10913, 10929, 10933, 10936, 10945, 10979, 11003, 11026, 11055, 11057, 11058, 11077, 11089, 11111, 11127, 11131, 11179, 11211, 11215, 11232, 11259, 11261, 11267, 11294, 11324, 11346, 11394, 11398, 11402, 11411, 11437, 11465, 11470, 11551, 11592, 11623, 11632, 11642, 11670, 11681, 11695, 11758, 11764, 11769, 11773, 11779, 11780, 11784, 11792, 11798, 11811, 11849, 11850, 11906, 11915, 11923, 11926, 11932, 11964, 11966, 11969, 12018, 12037, 12044, 12047, 12051, 12056, 12057, 12058, 12082, 12104, 12105, 12107, 12113, 12138, 12192, 12205, 12218, 12237, 12249, 12269, 12292, 12299, 12313, 12328, 12329, 12359, 12376, 12435, 12442, 12460, 12464, 12517, 12546, 12568, 12619, 12632, 12640, 12642, 12656, 12659, 12679, 12687, 12692, 12709, 12720, 12732, 12748, 12773, 12817, 12839, 12869, 12880, 12887, 12894, 12909, 12923, 12924, 12930, 12961, 12962, 12973, 12979, 12996, 12997, 12998, 12999, 13009, 13011, 13049, 13051, 13053, 13065, 13089, 13105, 13117, 13121, 13124, 13144, 13219, 13228, 13231, 13233, 13240, 13247, 13253, 13259, 13283, 13285, 13287, 13310, 13315, 13319, 13320, 13322, 13331, 13381, 13457, 13477, 13484, 13495, 13498, 13524, 13527, 13530, 13541, 13561, 13586, 13640, 13647, 13675, 13699, 13704, 13705, 13714, 13738, 13742, 13752, 13759, 13796, 13809, 13844, 13852, 13876, 13901, 13915, 13937, 13962, 13972, 13989, 14026, 14036, 14049, 14068, 14075, 14091, 14121, 14128, 14135, 14153, 14157, 14162, 14164, 14171, 14227, 14258, 14266, 14285, 14297, 14303, 14320, 14332, 14334, 14359, 14369, 14370, 14371, 14382, 14400, 14423, 14490, 14492, 14493, 14501, 14521, 14563, 14568, 14569, 14593, 14600, 14603, 14622, 14626, 14636, 14717, 14750, 14751, 14760, 14783, 14785, 14787, 14804, 14808, 14840, 14858, 14868, 14901, 14938, 14957, 14967, 14968, 14995, 15031, 15034, 15035, 15045, 15064, 15091, 15092, 15115, 15128, 15129, 15171, 15195, 15201, 15203, 15205, 15211, 15213, 15214, 15220, 15231, 15242, 15246, 15248, 15282, 15304, 15313, 15314, 15316, 15350, 15385, 15401, 15405, 15407, 15409, 15421, 15422, 15461, 15463, 15465, 15483, 15490, 15491, 15492, 15497, 15560, 15574, 15616, 15619, 15621, 15636, 15643, 15657, 15659, 15677, 15691, 15706, 15737, 15738, 15754, 15762, 15771, 15785, 15806, 15816, 15818, 15822, 15840, 15857, 15898, 15932, 15944, 15954, 16016, 16022, 16037, 16085, 16126, 16132, 16177, 16187, 16248, 16256, 16280, 16283, 16293, 16299, 16314, 16315, 16327, 16338, 16346, 16350, 16375, 16378, 16398, 16399, 16401, 16419, 16427, 16435, 16437, 16438, 16461, 16477, 16506, 16556, 16563, 16567, 16581, 16636, 16637, 16647, 16662, 16668, 16699, 16706, 16740, 16752, 16763, 16783, 16791, 16793, 16796, 16815, 16848, 16865, 16893, 16911, 16927, 16970, 16974, 16986, 17014, 17020, 17045, 17064, 17073, 17091, 17104, 17113, 17141, 17147, 17162, 17205, 17213, 17218, 17225, 17267, 17358, 17359, 17372, 17375, 17381, 17382, 17402, 17422, 17423, 17429, 17473, 17523, 17525, 17550, 17577, 17604, 17608, 17621, 17630, 17632, 17641, 17661, 17664, 17687, 17698, 17725, 17737, 17751, 17777, 17787, 17825, 17839, 17840, 17887, 17925, 17944, 17949, 17955, 17960, 17982, 18001, 18004, 18040, 18041, 18102, 18104, 18106, 18135, 18176, 18181, 18199, 18209, 18220, 18243, 18288, 18331, 18336, 18337, 18358, 18365, 18387, 18389, 18393, 18399, 18400, 18405, 18413, 18418, 18425, 18432, 18443, 18452, 18458, 18505, 18509, 18518, 18547, 18549, 18565, 18567, 18591, 18593, 18615, 18633, 18649, 18723, 18725, 18738, 18782, 18788, 18807, 18808, 18815, 18821, 18836, 18848, 18876, 18909, 18936, 18988, 18993, 18997, 19016, 19029, 19045, 19063, 19068, 19102, 19123, 19159, 19211, 19227, 19238, 19243, 19249, 19276, 19296, 19313, 19336, 19343, 19361, 19363, 19368, 19425, 19433, 19464, 19484, 19494, 19496, 19527, 19533, 19555, 19582, 19585, 19590, 19592, 19671, 19684, 19718, 19754, 19767, 19772, 19774, 19822, 19825, 19838, 19853, 19887, 19914, 19915, 19961, 19962, 19963, 19968, 19972, 19979, 19982, 20001, 20045, 20074, 20082, 20083, 20098, 20099, 20116, 20164, 20184, 20185, 20213, 20222, 20238, 20250, 20262, 20271, 20285, 20296, 20301, 20307, 20342, 20348, 20393, 20411, 20416, 20428, 20443, 20462, 20465, 20472, 20513, 20521, 20523, 20544, 20558, 20563, 20581, 20592, 20593, 20598, 20618, 20637, 20642, 20648, 20660, 20674, 20698, 20713, 20737, 20747, 20772, 20813, 20824, 20850, 20888, 20908, 20913, 20924, 20925, 20955, 20977, 20985, 20990, 20991, 20997, 21018, 21022, 21028, 21036, 21040, 21048, 21065, 21091, 21100, 21122, 21127, 21130, 21147, 21160, 21199, 21218, 21270, 21282, 21301, 21341, 21351, 21356, 21372, 21375, 21452, 21454, 21475, 21516, 21544, 21547, 21579, 21605, 21641, 21643, 21674, 21778, 21782, 21783, 21813, 21846, 21857, 21862, 21869, 21875, 21896, 21947, 21952, 21959, 21962, 21974, 21979, 21982, 22081, 22084, 22087, 22148, 22172, 22219, 22262, 22275, 22279, 22297, 22328, 22335, 22352, 22452, 22456, 22467, 22506, 22508, 22520, 22531, 22550, 22583, 22634, 22637, 22642]
drop_list=[]
for i in range(0,22656):
    if(i in drop):
        drop_list.append(0)
    if(i not in drop):
        drop_list.append(1)

drop_list2=[1]*22656


#device='cpu'
def get_activation(name):
    def hook(model, input, output):
        print(f"Hook activated for {name}")
        activation[name] = output.detach()
        print(f"Output shape: {output.detach().shape}")
    return hook
##这个函数的目的是用于调试和获取模型中指定层的激活值

activation = {} 

layer  = model.layer4[2]
layer_string='sub4_3'
print(type(layer))

hook = layer.register_forward_hook(get_activation(layer_string))

features = []
labelsV=[]
i=0
model.eval() 
with torch.no_grad():
    for data in testloader:
        images, labels = data     
        images = images.cuda()
        #labels = labels.to(device)
        outputs = model(images,drop_list)  # Extract features       
        featuresX = activation[layer_string]
        features.extend(featuresX.cpu().numpy())
        labelsV.extend(labels)

feat_array1=np.array(features);
mdic = {"feat_array1": feat_array1, 'labels':labelsV}   
#np.save('32.npy', feat_array)




hook.remove()



activation = {}

layer =  model.layer4[2]
layer_string='sub4_3'

hook = layer.register_forward_hook(get_activation(layer_string))

features = []
labelsV=[]
with torch.no_grad():
    for data in testloader:
        images, labels = data     
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images,drop_list2)  # Extract features
        featuresX = activation[layer_string]
        features.extend(featuresX.cpu().numpy())
        labelsV.extend(labels)
        
feat_array=np.array(features);
mdic = {"feat_array": feat_array, 'labels':labelsV}

#np.save('32.npy', feat_array)

hook.remove()


from sklearn.decomposition import PCA
import numpy as np
testDataFeatures = feat_array
testDataFeatures=testDataFeatures.reshape(17390,2048*4*4)
pca = PCA(n_components=400, svd_solver='arpack')
testDataFeatures = pca.fit_transform(testDataFeatures)
print(testDataFeatures.shape)
np.save('tstfeature_sub4_2_pca.npy', testDataFeatures)


from sklearn.decomposition import PCA
import numpy as np
testDataFeatures = feat_array1
testDataFeatures=testDataFeatures.reshape(17390,2048*4*4)
pca = PCA(n_components=400, svd_solver='arpack')
testDataFeatures = pca.fit_transform(testDataFeatures)
print(testDataFeatures.shape)
np.save('tstfeature_sub4_3_pca_pruning.npy', testDataFeatures)


correct = 0   # 定义预测正确的图片数，初始化为0
total = 0     # 总共参与测试的图片数，也初始化为0
ori_label=[]
pred_label=[]
torch.cuda.empty_cache()
model.eval()
with torch.no_grad():
    for data in testloader:  # 循环每一个batch
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        model.eval()  # 把模型转为test模式
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        outputs = model(images,drop_list2)  # 输入网络进行测试

        # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        _, predicted = torch.max(outputs.data, 1)
        predicted=predicted.to('cpu')
        labels = labels.to('cpu')
        lab=np.array(labels)
        ori_label.append(lab)
        predicted=np.array(predicted)
        predicted=list(predicted)
        pred_label.append(predicted)


pre=[]
for i in range(136):
    for data in pred_label[i]:
        pre.append(data)

lab=[]
for i in range(136):
    for data in ori_label[i]:
        lab.append(data)

print(len(lab))

tr_lab=np.array(pre)
np.save('test_pred_label.npy', tr_lab)

tr_lab=np.array(lab)
np.save('test_label.npy', tr_lab)