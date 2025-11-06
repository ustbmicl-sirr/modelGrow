import torch.nn as nn
import torch.nn.functional as F
from reparameterizer import RepUnit
from torch.nn import init


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, planes, stride=1, deploy=False):
        super(BasicBlock, self).__init__()
        self.feature1 = RepUnit(in_dim=planes, out_dim=planes, base_kernel_size=(3, 3), stride=stride, deploy=deploy)
        self.feature2 = RepUnit(in_dim=planes, out_dim=planes, base_kernel_size=(3, 3), stride=1, deploy=deploy)

    def forward(self, x):
        out = F.relu(self.feature2(F.relu(self.feature1(x))) + x)
        return out

    def switch_to_deploy(self):
        self.feature1.switch_to_deploy()
        self.feature2.switch_to_deploy()


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, planes, stride=1, deploy=False):
        super(Bottleneck, self).__init__()
        self.feature1 = nn.Conv2d(planes, planes//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//4)
        self.feature2 = RepUnit(in_dim=planes//4, out_dim=planes//4, base_kernel_size=(3, 3), stride=stride, deploy=deploy)
        self.feature3 = nn.Conv2d(planes//4, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.feature1(x)))
        out = F.relu(self.feature2(out))
        out = F.relu(self.bn3(self.feature3(out)) + x)
        return out

    def switch_to_deploy(self):
        self.feature2.switch_to_deploy()


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


class ResNet4Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, deploy=False):
        super(ResNet4Block, self).__init__()
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, deploy=deploy)
        self.layer2 = self._make_layer(block, 96, num_blocks[1], stride=1, deploy=deploy)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1, deploy=deploy)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1, deploy=deploy)
        self.downstream1 = Downstream(in_dim=64, out_dim=96)
        self.downstream2 = Downstream(in_dim=96, out_dim=128)
        self.downstream3 = Downstream(in_dim=128, out_dim=256)
        self.linear = nn.Linear(256, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, num_blocks, stride, deploy):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, stride, deploy=deploy))
        return nn.Sequential(*layers)

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
            if isinstance(net, BasicBlock) or isinstance(net, Bottleneck) or isinstance(net, Downstream):
                net.switch_to_deploy()
            else:
                for c in children:
                    foo(c)
        foo(self.eval())


class ResNet3Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, deploy=False):
        super(ResNet3Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, deploy=deploy)
        self.layer2 = self._make_layer(block, 96, num_blocks[1], stride=1, deploy=deploy)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1, deploy=deploy)
        self.downstream1 = Downstream(in_dim=64, out_dim=96)
        self.downstream2 = Downstream(in_dim=96, out_dim=128)
        self.linear = nn.Linear(128, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, num_blocks, stride, deploy):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, stride, deploy=deploy))
        return nn.Sequential(*layers)

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
            if isinstance(net, BasicBlock) or isinstance(net, Bottleneck) or isinstance(net, Downstream):
                net.switch_to_deploy()
            else:
                for c in children:
                    foo(c)
        foo(self.eval())


def get_ada_growing_basic_resnet(depths, num_classes=10, image_channels=3):
    if len(depths) == 3:
        return ResNet3Block(BasicBlock, depths, num_classes=num_classes, image_channels=image_channels)
    return ResNet4Block(BasicBlock, depths, num_classes=num_classes, image_channels=image_channels)


def get_ada_growing_bottleneck_resnet(depths, num_classes=10, image_channels=3):
    if len(depths) == 3:
        return ResNet3Block(Bottleneck, depths, num_classes=num_classes, image_channels=image_channels)
    return ResNet4Block(Bottleneck, depths, num_classes=num_classes, image_channels=image_channels)
