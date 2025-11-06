import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


def apply_2_4_sparsity(tensor):
    shape = tensor.shape
    flat_tensor = tensor.view(-1, 4)
    _, indices = torch.topk(torch.abs(flat_tensor), k=2, dim=1)
    mask = torch.zeros_like(flat_tensor).scatter_(1, indices, 1)
    return (flat_tensor * mask).view(shape)


def prune_conv_layer(layer, sparsity, use_2_4_sparsity=False):
    weights = layer.weight.data
    if use_2_4_sparsity:
        new_weights = apply_2_4_sparsity(weights)
    else:
        threshold = torch.quantile(torch.abs(weights), sparsity)
        mask = torch.abs(weights) > threshold
        new_weights = weights * mask.float()
    layer.weight.data = new_weights


def prune_cnn_model(model, sparsity, use_2_4_sparsity):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune_conv_layer(module, sparsity, use_2_4_sparsity)
            print(f"Pruned {name} using {'2:4 sparsity' if use_2_4_sparsity else 'sparsity'}.")


def test(model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define your CNN architecture here
        pass

    def forward(self, x):
        # Implement the forward pass
        pass

# 定义训练函数
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning for Pruned CNN Models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train and prune')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--sparsity', type=float, default=0.7, help='Sparsity level for pruning')
    parser.add_argument('--use_2_4_sparsity', action='store_true', help='Use 2:4 sparsity instead of regular pruning')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved sparse model')
    args = parser.parse_args()

    # Load and prepare CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path))

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_model(model, device, train_loader, optimizer, epoch)
        test(model, device)
        prune_cnn_model(model, args.sparsity, args.use_2_4_sparsity)
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
