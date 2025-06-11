# -*-coding: utf-8 -*-
import os, sys
sys.path.append(os.getcwd())
import onnx
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import math
import copy
import collections
import re
from onnx import helper
import sys,getopt
import pandas as pd
import onnxoptimizer
from teste1 import *
import onnxruntime as ort
from utils.func import *
from prune_method import *
import os
import argparse
from collections import Counter
import torchvision.models as models

parser = argparse.ArgumentParser(description='ONNX Training')
parser.add_argument('--save_prune_o_model_path', default="pytorch_checkpoint_cifar100/AlexNet/ckpt_cifar100.onnx", help='save_prune_o_model_path', type=str)
parser.add_argument('--data_root', default="/data/rdd/cifar100/", help='', type=str)

args = parser.parse_args()




def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def get_img_tensor_imagenet(img_path, use_cuda, get_size=False):
    img = Image.open(img_path).convert('RGB')
    #print(np.array(img).shape)
    if len(np.array(img).shape) != 3:
        img = img.convert('RGB')

    original_w, original_h = img.size
 
    #img_size = (32, 32)  # crop image to (224, 224)
    #img.thumbnail(img_size, Image.LANCZOS)
    #img = img.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if use_cuda:
        img_tensor = img_tensor.cuda()
    if get_size:
        return img_tensor, original_w, original_w
    else:
        return img_tensor

def test_onnx(model, data_root):
    
    if ort.get_device()=="CPU":
        config = ort.SessionOptions()
        ret,val=subprocess.getstatusoutput("cat /proc/cpuinfo | grep 'core id' |sort |uniq | wc -l")
        if ret==0:
            cpu_num_thread = int(val)
        else:
            cpu_num_thread=4
        config.intra_op_num_threads = cpu_num_thread
        config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers=["CPUExecutionProvider"]
        ort_session = ort.InferenceSession(model.SerializeToString(),providers=providers,sess_options=config)
    elif ort.get_device()=="GPU":
        providers=["GPUExecutionProvider"]
        ort_session = ort.InferenceSession(model.SerializeToString(),providers=providers)
    

    txt_file = os.path.join(data_root, "test.txt")

    images_path = []
    images_labels = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(' ')
            images_path.append(item[0])
            images_labels.append(item[1])
    #file_model = open('Cifar10_test1_onnx_feature.txt', 'w',encoding='utf-8')
    correct = 0
    total = 0
    for i in range(len(images_path)):
        #print(images_path[i])
        to_tensor = get_img_tensor_imagenet(data_root+images_path[i],False)
        #print(to_tensor)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(to_tensor)}
        #获取所有节点输出
        outputs = [x.name for x in ort_session.get_outputs()] #ort_session.get_outputs()[0].name是原模型的单一输出
        ort_outs = ort_session.run(output_names=outputs, input_feed=ort_inputs) 
        
        preds = ort_outs[0]
        indexes = np.argsort(preds[0])[::-1]
        total = total + 1
        #print(indexes[0])
        if str(indexes[0]) == str(images_labels[i]):
            correct = correct + 1
        print(total, ":", correct/total)
    print('###########################')
    print(correct/total)
    #print()




model = loadOnnxModel(args.save_prune_o_model_path)
test_onnx(model, args.data_root)
    
print('sucess')










