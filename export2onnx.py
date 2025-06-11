import torch.nn as nn
import torch.nn.init as init
import torch
import argparse
import os
from models import *


def parse_args():
    """Parse input arguments."""
    desc = "HCW model zoo: image classification train"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', default='ResNet18', help='choice a model to train and eval. eg: alenet, vgg16',
                        type=str)
    parser.add_argument('--model_path', default="checkpoint/ResNet18/ckpt.pth", help='path to model', type=str)
    parser.add_argument('--num_class', default=10, help='number of class', type=int)
    parser.add_argument('--export_name', default="checkpoint/ResNet18/ckpt.onnx", help='path to onnx model', type=str)
    parser.add_argument('--input_size', default=32, help='path to onnx model', type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    num_class = args.num_class

    batch_size = 1  # just a random number
    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    net = None
    # Model
    print('==> Building model..')
    if args.model_name == "VGG16":
        net = VGG('VGG16')
    elif args.model_name == "ResNet18":
        net = ResNet18()
    elif args.model_name == "ResNet34":
        net = ResNet34()
    elif args.model_name == "ResNet50":
        net = ResNet50()
    elif args.model_name == "PreActResNet18":
        net = PreActResNet18()
    elif args.model_name == "GoogLeNet":
        net = GoogLeNet()
    elif args.model_name == "DenseNet121":
        net = DenseNet121()
    elif args.model_name == "ResNeXt29_2x64d":
        net = ResNeXt29_2x64d()
    elif args.model_name == "MobileNet":
        net = MobileNet()
    elif args.model_name == "MobileNetV2":
        net = MobileNetV2()
    elif args.model_name == "DPN92":
        net = DPN92()
    elif args.model_name == "ShuffleNetG2":
        net = ShuffleNetG2()
    elif args.model_name == "SENet18":
        net = SENet18()
    elif args.model_name == "ShuffleNetV2":
        net = ShuffleNetV2(1)
    elif args.model_name == "EfficientNetB0":
        net = EfficientNetB0()
    elif args.model_name == "RegNetX_200MF":
        net = RegNetX_200MF()
    elif args.model_name == "SimpleDLA":
        net = SimpleDLA()
    elif args.model_name == "LeNet":
        net = LeNet()
    elif args.model_name == "LeNetv2":
        net = LeNet_v2()
    elif args.model_name == "LeNetv3":
        net = LeNet_v3()
    elif args.model_name == "AlexNet":
        net = AlexNet()
    elif args.model_name == "AlexNetv2":
        net = AlexNetv2()
    else:
        print("Error: model name not define! Exit...")
        exit(1)
    net.load_state_dict(torch.load(args.model_path, map_location=map_location))

    # set the model to inference mode
    net.eval()

    # Input to the model
    x = torch.randn(batch_size, 3, args.input_size, args.input_size, requires_grad=True)
    torch_out = net(x)

    # Export the model
    # tmp = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      args.export_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=18,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    print("Done!")
