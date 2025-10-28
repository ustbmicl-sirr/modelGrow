import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from reparameterizer import RepUnit


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.expand_size =  max(in_size // reduction, 8)
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MobileNetBlock(nn.Module):
    def __init__(self, kernel_size, size, deploy=False):
        super(MobileNetBlock, self).__init__()
        expand_size = 4 * size
        self.conv1 = nn.Conv2d(size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = RepUnit(expand_size, expand_size, (kernel_size, kernel_size), groups=expand_size)
        self.act2 = nn.SiLU(inplace=True)
        self.se = SeModule(expand_size)
        self.conv3 = nn.Conv2d(expand_size, size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(size)
        self.act3 = nn.SiLU(inplace=True)

    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.conv2(out))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        return self.act3(out + skip)
    
    def switch_to_deploy(self):
        self.conv2.switch_to_deploy()


class Downstream(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Downstream, self).__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.downstream = RepUnit(in_dim, out_dim, (3, 3), stride=2, deploy=False)

    def forward(self, x):
        out = self.norm(x)
        out = self.downstream(x)
        return out
    
    def switch_to_deploy(self):
        self.downstream.switch_to_deploy()
  

class GrowingMobileNet4Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, deploy=False):
        super(GrowingMobileNet4Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, num_blocks[0], kernel_size=3, size=16, deploy=deploy)
        self.layer2 = self._make_layer(block, num_blocks[1], kernel_size=5, size=32, deploy=deploy)
        self.layer3 = self._make_layer(block, num_blocks[2], kernel_size=5, size=64, deploy=deploy)
        self.layer4 = self._make_layer(block, num_blocks[3], kernel_size=5, size=96, deploy=deploy)
        self.downstream1 = Downstream(in_dim=16, out_dim=32)
        self.downstream2 = Downstream(in_dim=32, out_dim=64)
        self.downstream3 = Downstream(in_dim=64, out_dim=96)
        self.linear = nn.Linear(96, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, kernel_size, size, deploy):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(kernel_size, size, deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.downstream1(out)
        out = self.layer2(out)
        out = self.downstream2(out)
        out = self.layer3(out)
        out = self.downstream3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        def foo(net):
            children = list(net.children())
            if isinstance(net, MobileNetBlock) or isinstance(net, Downstream):
                net.switch_to_deploy()
            else:
                for c in children:
                    foo(c)
        foo(self.eval())


class GrowingMobileNet3Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, deploy=False):
        super(GrowingMobileNet3Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, num_blocks[0], kernel_size=3, size=32, deploy=deploy)
        self.layer2 = self._make_layer(block, num_blocks[1], kernel_size=5, size=64, deploy=deploy)
        self.layer3 = self._make_layer(block, num_blocks[2], kernel_size=5, size=96, deploy=deploy)
        self.downstream1 = Downstream(in_dim=32, out_dim=64)
        self.downstream2 = Downstream(in_dim=64, out_dim=96)
        self.linear = nn.Linear(96, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, kernel_size, size, deploy):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(kernel_size, size, deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.downstream1(out)
        out = self.layer2(out)
        out = self.downstream2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        def foo(net):
            children = list(net.children())
            if isinstance(net, MobileNetBlock) or isinstance(net, Downstream):
                net.switch_to_deploy()
            else:
                for c in children:
                    foo(c)
        foo(self.eval())


def get_ada_growing_mobilenetv3(depths, num_classes=10, image_channels=3, deploy=False):
    if len(depths) == 3:
        return GrowingMobileNet3Block(MobileNetBlock, depths, num_classes=num_classes, image_channels=image_channels, deploy=deploy)
    return GrowingMobileNet4Block(MobileNetBlock, depths, num_classes=num_classes, image_channels=image_channels, deploy=deploy)
