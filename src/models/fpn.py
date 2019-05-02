'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # увеличить до 5*5 – не сходится!
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)

        # Upsample using convolution transpose
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)        
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.predict1 = nn.Sequential(
                        nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(), # change to the ReLU
                    )
        # kernel_size=1
        self.predict2 = nn.Sequential(
                        nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
                        nn.Sigmoid(), # change to the ReLU
                    )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        ~original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        # conv transpose with weights that was used for shrinking th
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        # smooth of the input
        #x = smooth(3, 3)(x)
        #c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.relu(self.conv1(x))
        #c1 = F.max_pool2d(c1, kernel_size=3, stride=1, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self.upsample4(p5, output_size=c4.size()) + self.latlayer1(c4)  # self._upsample_add(p5, self.latlayer1(c4))
        p3 = self.upsample3(p4, output_size=c3.size()) + self.latlayer2(c3)  # self._upsample_add(p4, self.latlayer2(c3))
        p2 = self.upsample2(p3, output_size=c2.size()) + self.latlayer3(c2)  # self._upsample_add(p3, self.latlayer3(c2))
        p1 = self.upsample1(p2, output_size=c1.size()) + self.latlayer4(c1)  # self._upsample_add(p2, self.latlayer4(c1))
  
        p5 = self.predict2(p5)
        p4 = self.predict2(p4)
        p3 = self.predict2(p3)
        p2 = self.predict2(p2)
        p1 = self.predict2(p1)
        return p1, p2, p3, p4, p5


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,576,576)))
    for fm in fms:
        print(fm.size())

#test()