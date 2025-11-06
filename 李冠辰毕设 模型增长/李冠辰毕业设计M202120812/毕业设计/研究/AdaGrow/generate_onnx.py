import argparse
import torch
import torch.nn as nn
import torch.onnx
import models
import timm


def get_model(model_name: str, arch: list, args):
    try:
        net = timm.create_model(model_name, 
                                num_classes=args.num_classes,
                                in_chans=args.image_channels)
        if "resnet" in model_name:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    except:
        if "vit" in model_name:
            net = getattr(models, model_name)(depth=arch[0], 
                                              heads=arch[1], 
                                              num_classes=args.num_classes, 
                                              image_channels=args.image_channels)
        else:
            net = getattr(models, model_name)(depths=arch, 
                                              num_classes=args.num_classes, 
                                              image_channels=args.image_channels)
    print(net)
    net = net.eval()
    return net


def export_onnx(model, dummy_input, save_path="simple_mlp.onnx"):
    torch.onnx.export(model,
                      dummy_input,
                      save_path,
                      export_params=True,
                      opset_version=10,             # ONNX版本
                      do_constant_folding=True,     # 是否执行常量折叠优化
                      input_names=['input'],        # 输入tensor的名字
                      output_names=['output'],      # 输出tensor的名字
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model_name', type=str, default="get_runtime_basic_resnet", help='Model name to use for export')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the model')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels in the input image')
    parser.add_argument('--input_height', type=int, default=32, help='Input image height')
    parser.add_argument('--input_width', type=int, default=32, help='Input image width')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the dynamic axes')
    parser.add_argument('--arch', nargs='+', type=int, default=[], help='Architecture list for the model, if applicable')
    parser.add_argument('--output', type=str, default='auto', help='Path to save the ONNX model')
    args = parser.parse_args()

    model = get_model(args.model_name, args.arch, args)
    dummy_input = torch.randn(args.batch_size, args.image_channels, args.input_height, args.input_width)
    if args.output == "auto":
        str_result = '-'.join(map(str, args.arch)) if len(args.arch) > 0 else ""
        args.output = args.model_name + "_" + str_result + ".onnx"
    export_onnx(model, dummy_input, args.output)
    
    print(f"Model exported to {args.output} successfully.")