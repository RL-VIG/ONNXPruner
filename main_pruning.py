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
import time

parser = argparse.ArgumentParser(description='ONNX Training')
parser.add_argument('--o_model_path',  default="pytorch_checkpoint_cifar10/VGG16/ckpt_cifar10.onnx", help='o_model_path', type=str)
parser.add_argument('--save_prune_o_model_path', default="pytorch_checkpoint_cifar10/VGG16/ckpt_cifar10_prune_l1_src_0.5.onnx", help='save_prune_o_model_path', type=str)
parser.add_argument('--dataset', default="/data/rdd/cifar10_dataset/", help='dataset', type=str)
parser.add_argument('--prune_methon', default="l1_src", help='prune_methon', type=str)
args = parser.parse_args()


prune_node_list = {'Conv': 0.1}


output_limit = 100


prune_TAG = {'next':['Relu', 'MaxPool', 'Flatten'], 
             'stop':['Conv', 'Gemm'], 
             'last': ['Sum']}

Next_OP_list = ['Relu', 'MaxPool', 'AveragePool', 'Concat', 'Add', 'GlobalAveragePool', 'Flatten', 'LRN', 'Pad'] #不需要处理的算子，直接跳过

Special_OP_list = ['Concat', 'Add'] #需要特殊处理的算子，
#(1) 如果遇到concat算子，先查询该算子位于第几个feature, 第0个，直接剪枝，第n个，就需要f1_num+...+fn-1_num+id
#(2) 如果遇到sum算子，查看以这个sum节点为叶子节点的其他子节点，如果剪枝率不同，该子树不剪枝，如果相同，则比较fn+1*fn1+fn+1*fn2

Stop_OP_list = ['Conv', 'ConvTranspose', 'Gemm'] 

Stop_prune_OP_list = ['Conv', 'ConvTranspose', 'Gemm']

#Last_OP_list = ['Sum']

TAG = 'netx' # 'last','stop'

def model_prune_node(o_model_path, prune_node_list):
    model = loadOnnxModel(o_model_path)
    model.opset_import[0].version = 18

    graph = model.graph 
    node = graph.node 
    initializer_list = graph.initializer
    #对节点的剪枝率进行重置
    prune_node_list = init_note_rate(node, prune_node_list)
    #将剪枝的节点编号和剪枝率放进字典中
    prune_node_id_dct = find_prune_node_id(node, prune_node_list)

    #print("==========prune_node_id_dct==============", prune_node_id_dct)

    tree_node_dct = {} #存储需要剪枝的节点id，以及该节点的先序遍历id列表
    tree_node_leaf_dct = {} #存储需要剪枝的节点id，以及该节点构建tree的叶子节点列表

    #遍历prune_node_id_list，获取每个节点的子树
    for node_id in prune_node_id_dct.keys():
        list_node_input_tree, node_tree = dsf_find_new(node, node_id, Next_OP_list, Stop_OP_list)
        leaf_nodes = find_leaf_nodes(node_tree)#查找叶子节点

        list_node_input_tree.append('N') #插入该树是否处理标识符，Y表示处理，N表示未处理
        tree_node_dct[node_id] = list_node_input_tree
        tree_node_leaf_dct[node_id] = leaf_nodes
    #print("==========tree_node_dct==============", tree_node_dct)

    #遍历每个节点的tree，判断该节点中是否包含特殊处理的算子，如果包含，放入特殊处理字典
    tree_Special_op = {}
    for node_id in tree_node_dct.keys():
        tree_child_list = tree_node_dct[node_id][0:-1]
        
        tree_child_id_list = []
        for tree_child_id in tree_child_list:
            if node[tree_child_id].op_type in Special_OP_list:
                tree_child_id_list.append(str(tree_child_id)+str(node[tree_child_id].op_type))
        if len(tree_child_id_list) != 0:
            tree_Special_op[node_id] = tree_child_id_list
    #print("==========tree_Special_op==============", tree_Special_op)

    #开始执行剪枝
    for node_id in tree_node_dct.keys():
        tree_child_list = tree_node_dct[node_id]
        #print("==========tree_child_list==============", tree_child_list)
        i_num = 0
        for opid in tree_child_list[0:-1]:
            _optype = node[opid].op_type
            if _optype == 'Gemm':
                tree_child_list[-1] = "Y"
            if _optype == 'GlobalAveragePool':
                tree_child_list[-1] = "Y"
            if _optype == 'Add':
                i_num = i_num + 1
        if tree_child_list[-1] == "Y":
            continue
        num_channet_sum = 0

        #Add算子中第二个算子的相应参数
        prune_nodel_num2_init = None
        next_node_num2_init = None
        prune_rate2_init = None
        #判断是否为特殊处理的tree，如果是，执行特殊处理算子
        #print(tree_Special_op)
        #print(node_id)
        if node_id in tree_Special_op:
            #取出该节点special_op的value, 并将tree_node_id和tree_child_list存入this_Special_op_list中
            this_Special_op_list = {}
            #this_Special_op_list[node_id] = tree_child_list[0:-1]
            Special_op_value = tree_Special_op[node_id]
            #print('Special_op_value', Special_op_value)
            #print('---------')
            #寻找与Special_op_value相同的，并放进this_Special_op_list
            for key in tree_Special_op.keys():
                list_U = list(set(Special_op_value) & set(tree_Special_op[key]))
                #print('list_U', list_U)
                if len(list_U) != 0:
                    this_Special_op_list[key] = tree_node_dct[key][0:-1]
            #print('this_Special_op_list', this_Special_op_list)
            #print('---------')
            #print(Special_op_value)
            #特殊算子的类型
            if Special_op_value[-1][-6:] == 'Concat': 
                #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', Special_op_value)
                num_channet_sum = 0
                for key in this_Special_op_list.keys():
                    #print(key)
                    if node_id == key:
                        break
                    if node_id != key:
                        for order_j in range(len(initializer_list)):
                            if(initializer_list[order_j].name==node[key].input[1]):
                                num_channet_sum = num_channet_sum + initializer_list[order_j].dims[0]
            if Special_op_value[-1][-3:] == 'Add':
                #print('===============this_Special_op_list===================', this_Special_op_list)
                if len(list(this_Special_op_list.keys())) == 1:
                    node_id = list(this_Special_op_list.keys())[0]
                    tree_node_dct[node_id][-1] = 'Y'
                    l1_src_prune(node_id, tree_node_leaf_dct[node_id], node, initializer_list, prune_node_id_dct[node_id], num_channet_sum)
                    continue
                if len(list(this_Special_op_list.keys())) == 2:
                    node_id = list(this_Special_op_list.keys())[0]

                    node_id1 = list(this_Special_op_list.keys())[1]
                    prune_nodel_num1_init = node_id1
                    next_node_num1_init = tree_node_leaf_dct[node_id1]
                    prune_rate1_init = prune_node_id_dct[node_id1]

                    tree_node_dct[node_id][-1] = 'Y'
                    tree_node_dct[node_id1][-1] = 'Y'
                    l1_src_prune(node_id, tree_node_leaf_dct[node_id], node, initializer_list, prune_node_id_dct[node_id], num_channet_sum,
                             prune_nodel_num1 = prune_nodel_num1_init, next_node_num1 = next_node_num1_init, prune_rate1 = prune_rate1_init)
                    continue
        
                
                if len(list(this_Special_op_list.keys())) == 3:
                    node_id = list(this_Special_op_list.keys())[0]
                    
                    node_id1 = list(this_Special_op_list.keys())[1]
                    prune_nodel_num1_init = node_id1
                    next_node_num1_init = tree_node_leaf_dct[node_id1]
                    prune_rate1_init = prune_node_id_dct[node_id1]

                    node_id2 = list(this_Special_op_list.keys())[2]
                    prune_nodel_num2_init = node_id2
                    next_node_num2_init = tree_node_leaf_dct[node_id2]
                    prune_rate2_init = prune_node_id_dct[node_id2]

                    tree_node_dct[node_id][-1] = 'Y'
                    tree_node_dct[node_id1][-1] = 'Y'
                    tree_node_dct[node_id2][-1] = 'Y'
                    #print(tree_node_leaf_dct[node_id])

                    l1_src_prune(node_id, tree_node_leaf_dct[node_id], node, initializer_list, prune_node_id_dct[node_id], num_channet_sum,
                             prune_nodel_num1 = prune_nodel_num1_init, next_node_num1 = next_node_num1_init, prune_rate1 = prune_rate1_init,
                             prune_nodel_num2 = prune_nodel_num2_init, next_node_num2 = next_node_num2_init, prune_rate2 = prune_rate2_init)
                    continue
        #print('num_channet_sum', num_channet_sum)
        l1_src_prune(node_id, tree_node_leaf_dct[node_id], node, initializer_list, prune_node_id_dct[node_id], num_channet_sum)
        tree_node_dct[node_id][-1] = 'Y'
        #print('Done!')
    onnx.checker.check_model(model)
    return model
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
        #print(total, ":", correct/total)
    #print('###########################')
    print(correct/total)
    #print()


model = model_prune_node(args.o_model_path, prune_node_list)

onnx.save(model,args.save_prune_o_model_path)

test_onnx(model, "/data/rdd/cifar10_dataset/")
    
print('sucess')










