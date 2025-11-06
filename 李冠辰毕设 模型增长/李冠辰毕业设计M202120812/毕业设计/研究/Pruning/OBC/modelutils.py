import timm
import torch
import torch.nn as nn
from quant import *
import models
import torch.optim as optim


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear, ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for x, y in dataloader:
        preds.append(torch.argmax(model(x.to(dev)), 1))
        ys.append(y.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('%.2f' % acc)
    if model.training:
        model.train()


def train(model, trainloader, epochs=50, learning_rate=0.005):
    model.train()  # 将模型设置为训练模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):  # 添加外循环以遍历多个epoch
        if epoch >= 20: 
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        if epoch >= 35: 
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        total_loss = 0.0
        for x, y in trainloader:
            x = x.to(device)  # 确保数据和模型在同一设备
            y = y.to(device)
            
            optimizer.zero_grad()  # 清零梯度
            output = model(x)  # 前向传播
            loss = loss_fn(output, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失

        average_loss = total_loss / len(trainloader)  # 计算平均损失
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {average_loss:.4f}')

def run(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if loss:
        return nn.functional.cross_entropy(out, batch[1].to(dev)).item() * batch[0].shape[0]
    return out


def get_model(model_name, arch):
    try:
        net = timm.create_model(model_name, num_classes=10, in_chans=3)
        if "resnet" in model_name:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    except:
        if "vit" in model_name:
            try:
                net = getattr(models, model_name)(depth=arch[0], heads=arch[1], num_classes=10, image_channels=3)
            except:
                net = getattr(models, model_name)(num_classes=10, image_channels=3)
        else:
            net = getattr(models, model_name)(depths=arch, num_classes=10, image_channels=3)
    print(net)
    net = net.to(DEV)
    net = net.eval()
    return net


def get_functions():
    return lambda: get_model, test, run