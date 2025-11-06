import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Model Pruning')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity level for pruning (e.g., 0.7 for 70%)')
    parser.add_argument('--use_2_4_sparsity', action='store_true', help='Apply 2:4 sparsity instead of regular pruning')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model state')
    parser.add_argument('--save_path', type=str, help='Path to save the pruned model state (optional)')
    args = parser.parse_args()

    state = torch.load(args.model_path)
    model = state['net']
    prune_cnn_model(model, args.sparsity, args.use_2_4_sparsity)
    test(model, "cuda:0")

    if args.save_path:
        torch.save({
            'net': model,
            'acc': state.get('acc', None), 
            'epoch': state.get('epoch', None) 
        }, os.path.join(args.save_path, 'pruned_ckpt.pth'))
        print(f"Pruned model saved to {os.path.join(args.save_path, 'pruned_ckpt.pth')}")