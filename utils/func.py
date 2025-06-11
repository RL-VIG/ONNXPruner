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





#加载模型函数
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

#查找节点的直接关联点
def find_relevance_node(node, prune_nodel_num):
    list_input = []
    for j in range(len(node)):
        #print(node[prune_nodel_num].output[0])
        #print(node[j])
        #print('------------------------------------')
        if node[prune_nodel_num].output[0] == node[j].input[0]:#查找所有以该节点输出做为输入的节点，放进list中
            #print(j)
            list_input.append(j)
    return list_input


#根据key(节点名称找到节点id)和该节点的剪枝率
def find_node_id(node, key):
    prune_channel_num = None
    for i in range(len(node)):#遍历模型节点
            if(node[i].name==key):#查找节点名称进行剪枝
                prune_nodel_num=i
    return prune_nodel_num

#以这些节点做为起始节点，构建一个链路并存储
def find_node_chain(node_num, node):
    #print(node_num)
    list_node_chain = []
    list_node_chain.append(node_num)
    #起始节点 节点类型为：node[node_num].op_type 输出为 node[node_num].input[1]，当op_type为next时候，继续查找，当为stop时候，剪裁并停止
    current_TAG = ''
    current_num = node_num
    
    if node[node_num].op_type == 'Relu' or node[node_num].op_type == 'MaxPool':
        current_TAG = 'next'
    
    if node[node_num].op_type in Stop_OP_list:
        current_TAG = 'stop'
        list_node_chain.append(node_num)
    
    #print('current_TAG', current_TAG)
    while current_TAG == 'next':
        for k in range(len(node)):
            if node[k].input[0] == node[current_num].output[0]:
                current_num = k
                if node[current_num].op_type in Next_OP_list: #if (4 in test_list):
                    current_TAG = 'next'
                    list_node_chain.append(current_num)
                if node[current_num].op_type in Stop_OP_list:
                    current_TAG = 'stop'
                    list_node_chain.append(current_num)
        #print(current_TAG)
    return list_node_chain

def prune_node(prune_nodel_num, node, initializer_list, prune_rate):
    for j in range(len(initializer_list)):
        if(initializer_list[j].name==node[prune_nodel_num].input[1]):
            # 解码获得权重参数
            #print(initializer_list[j].name)
            wight = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
            #print(wight)
            #print(onnx.numpy_helper.to_array(initializer_list[j]))
            wigth_np = wight.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
            #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
            #print(wigth_np.shape)
            #对权重矩阵进行重要性评估
            wigth_np_copy = np.abs(wigth_np)
            L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
            arg_max = np.argsort(L1_norm)
            #arg_max_rev = arg_max[::-1]
            prune_channel_num = arg_max[0 : round(prune_rate * len(arg_max))]
            #对重要性较小的几个通道进行剪裁
            wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=0)
            #print(prune_channel_num)
            #重新赋值initializer
            #print(initializer_list[j].float_data)
            initializer_list[j].raw_data = wigth_np_prune.tobytes()
            initializer_list[j].dims[0] = wigth_np_prune.shape[0]
            #print(initializer_list[j].dims[0])

        if len(node[prune_nodel_num].input)>2: #判断该节点是否存在偏置,如果存在，把偏置也剪裁掉
            if(initializer_list[j].name==node[prune_nodel_num].input[2]):
                bias = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                #bias = onnx.numpy_helper.to_array(initializer_list[j])

                bias_prune =  np.delete(bias, prune_channel_num)
                #重新赋值initializer
                initializer_list[j].raw_data = bias_prune.tobytes()
                initializer_list[j].dims[0] = bias_prune.shape[0]
    return prune_channel_num

def prune_nextnode(prune_channel_num, node, node_num, initializer_list):
    for current_j in range(len(initializer_list)):
        #print(initializer_list[current_j].name)
        
        if(initializer_list[current_j].name==node[node_num].input[1]):
            #print(node[current_num].input[1])
            #print(initializer_list[j].name)
            wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
            #print(initializer_list[current_j].dims)
            wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
            #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
            wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=1)
            #重新赋值initializer
            initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
            initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]

def my_prune(prune_nodel_num, next_node_num, node, initializer_list, prune_rate):
    #weight_l1 = []
    #weight_next_l1 = []
    for j in range(len(initializer_list)):
        if(initializer_list[j].name==node[prune_nodel_num].input[1]):
            # 解码获得权重参数
            #print(initializer_list[j].name)
            wight = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
            #print(wight)
            #print(onnx.numpy_helper.to_array(initializer_list[j]))
            wigth_np = wight.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
            #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
            #print(wigth_np.shape)
            #对权重矩阵进行重要性评估 L1-------------------
            wigth_np_copy = np.abs(wigth_np)
            L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
            #l2_norm = np.sqrt(np.sum(wigth_np_copy*wigth_np_copy, axis=(1, 2, 3)))
            
            #print(l2_norm.shape)
            nood_norm = L1_norm

            #print(np.linalg.norm(X, axis=0))
            #weight_l1 = L1_norm
            #print(initializer_list[j].dims[0])
        if(initializer_list[j].name==node[next_node_num].input[1]):
            if node[next_node_num].op_type == "Conv":
                # 解码获得权重参数
                #print(initializer_list[j].name)
                wight_n = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                #print(wight)
                #print(onnx.numpy_helper.to_array(initializer_list[j]))
                wigth_np_n = wight_n.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
                #print(wigth_np.shape)
                #对权重矩阵进行重要性评估
                wigth_np_copy_n = np.abs(wigth_np_n)
                L1_norm_n = np.sum(wigth_np_copy_n, axis=(0, 2, 3))
                #l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0, 2, 3)))
                nood_next_norm = L1_norm_n
                #weight_next_l1 = L1_norm_n
                #print(initializer_list[j].dims[0])
            if node[next_node_num].op_type == "Gemm":
                # 解码获得权重参数
                #print(initializer_list[j].name)
                wight_n = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                #print(wight)
                #print(onnx.numpy_helper.to_array(initializer_list[j]))
                wigth_np_n = wight_n.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
                #print(wigth_np.shape)
                #对权重矩阵进行重要性评估
                wigth_np_copy_n = np.abs(wigth_np_n)
                L1_norm_n = np.sum(wigth_np_copy_n, axis=(0))
                #l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0)))
                nood_next_norm = L1_norm_n
                #weight_next_l1 = L1_norm_n
                #print(initializer_list[j].dims[0])
    weight_Tree = nood_norm * nood_next_norm

    for j in range(len(initializer_list)):
        if(initializer_list[j].name==node[prune_nodel_num].input[1]):
            
            arg_max = np.argsort(weight_Tree)
            #arg_max_rev = arg_max[::-1]
            prune_channel_num = arg_max[0 : round(prune_rate * len(arg_max))]
            #对重要性较小的几个通道进行剪裁
            wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=0)
            #print(prune_channel_num)
            #重新赋值initializer
            #print(initializer_list[j].float_data)
            initializer_list[j].raw_data = wigth_np_prune.tobytes()
            initializer_list[j].dims[0] = wigth_np_prune.shape[0]
            #print(initializer_list[j].dims[0])

        if len(node[prune_nodel_num].input)>2: #判断该节点是否存在偏置,如果存在，把偏置也剪裁掉
            if(initializer_list[j].name==node[prune_nodel_num].input[2]):
                bias = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                #bias = onnx.numpy_helper.to_array(initializer_list[j])

                bias_prune =  np.delete(bias, prune_channel_num)
                #重新赋值initializer
                initializer_list[j].raw_data = bias_prune.tobytes()
                initializer_list[j].dims[0] = bias_prune.shape[0]
    #print(node[prune_nodel_num].name,':',prune_channel_num)

    for current_j in range(len(initializer_list)):
        #print(initializer_list[current_j].name)
        
        if(initializer_list[current_j].name==node[next_node_num].input[1]):
            if node[next_node_num].op_type == "Conv":
                #print(node[current_num].input[1])
                #print(initializer_list[j].name)
                wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                #print(initializer_list[current_j].dims)
                wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
                wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=1)
                #重新赋值initializer
                initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
            if node[next_node_num].op_type == "Gemm":
                #print(node[current_num].input[1])
                #print(initializer_list[j].name)
                wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                #print(initializer_list[current_j].dims)
                wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
                wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=1)
                #重新赋值initializer
                initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]

    #print()
def src_prune(prune_nodel_num, next_node_num, node, initializer_list, prune_rate):
    #weight_l1 = []
    #weight_next_l1 = []
    for j in range(len(initializer_list)):
        if(initializer_list[j].name==node[prune_nodel_num].input[1]):
            # 解码获得权重参数
            #print(initializer_list[j].name)
            wight = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
            #print(wight)
            #print(onnx.numpy_helper.to_array(initializer_list[j]))
            wigth_np = wight.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
            #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
            #print(wigth_np.shape)
            #对权重矩阵进行重要性评估
            wigth_np_copy = np.abs(wigth_np)
            L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
            #l2_norm = np.sqrt(np.sum(wigth_np_copy*wigth_np_copy, axis=(1, 2, 3)))
            
            #print(l2_norm.shape)
            nood_norm = L1_norm
            #weight_l1 = L1_norm
            #print(initializer_list[j].dims[0])
    weight_src = nood_norm

    for j in range(len(initializer_list)):
        if(initializer_list[j].name==node[prune_nodel_num].input[1]):
            
            arg_max = np.argsort(weight_src)
            #arg_max_rev = arg_max[::-1]
            prune_channel_num = arg_max[0 : round(prune_rate * len(arg_max))]
            #对重要性较小的几个通道进行剪裁
            wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=0)
            #print(prune_channel_num)
            #重新赋值initializer
            #print(initializer_list[j].float_data)
            initializer_list[j].raw_data = wigth_np_prune.tobytes()
            initializer_list[j].dims[0] = wigth_np_prune.shape[0]
            #print(initializer_list[j].dims[0])

        if len(node[prune_nodel_num].input)>2: #判断该节点是否存在偏置,如果存在，把偏置也剪裁掉
            if(initializer_list[j].name==node[prune_nodel_num].input[2]):
                bias = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                #bias = onnx.numpy_helper.to_array(initializer_list[j])

                bias_prune =  np.delete(bias, prune_channel_num)
                #重新赋值initializer
                initializer_list[j].raw_data = bias_prune.tobytes()
                initializer_list[j].dims[0] = bias_prune.shape[0]
    #print(node[prune_nodel_num].name,':',prune_channel_num)

    for current_j in range(len(initializer_list)):
        #print(initializer_list[current_j].name)
        
        if(initializer_list[current_j].name==node[next_node_num].input[1]):
            if node[next_node_num].op_type == "Conv":
                #print(node[current_num].input[1])
                #print(initializer_list[j].name)
                wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                #print(initializer_list[current_j].dims)
                wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
                wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=1)
                #重新赋值initializer
                initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
            if node[next_node_num].op_type == "Gemm":
                #print(node[current_num].input[1])
                #print(initializer_list[j].name)
                wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                #print(initializer_list[current_j].dims)
                wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
                wigth_np_prune = np.delete(wigth_np, prune_channel_num, axis=1)
                #重新赋值initializer
                initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]

    #print()
def get_path(start, child_father_dict):
    """
    从叶子节点出发，不断向根节点遍历，得到所有叶子节点到根节点的路径
    :param start:
            eg: 'c'
    :param child_father_dict:
            eg: {'c': {'d', 'b'}, 'd': {'a'}, 'b': {'a'}}
    :return: all_path:
            eg: [[['c', 'b', 'a'], ['c', 'd', 'a']]]
    """

    def _dfs(node, path, res):
        path.append(node)
        if node not in child_father_dict:
            res.append(path.copy())
            path.pop()
            return
        for ni in child_father_dict[node]:
            _dfs(ni, path, res)
        path.pop()

    all_path = []
    if start in child_father_dict:
        _dfs(start, [], all_path)
    return all_path

def query_id_optype(node):
    _op_type_dct = {}
    for k in range(len(node)):
        _op_type = node[k].op_type
        _op_type_dct[k] = _op_type


def dsf_find(node, prune_nodel_num, Next_OP_list, Stop_OP_list):

    #name = str(prune_nodel_num) + '_node'
    #_node_0 = BinaryTree(prune_nodel_num)
    #rl = r.insertleft('b')
    #rr = r.insertright('c')
    #print('先序遍历为:')
    #r.preorder()


    #构建树的根节点，并赋值
    node_0 = BinaryTree()
    node_0.setrootvalue(prune_nodel_num)
    
    list_yezi = []
    list_yezi.append(prune_nodel_num)
    #print(list_yezi)
    list_tree_node = []
    list_tree_node.append(node_0)


    def dsf(list_yezi, list_tree_node):
        #if len(list_yezi) == 1:
            #rl = rl.insertleft(list_yezi[0])
        list_yezi_ebm = []
        list_tree_node_ebm = []
        
        for yezi,tree_node in zip(list_yezi, list_tree_node):
            #print(yezi)
            list_ebm = []
            list_node_ebm = []
            for k in range(len(node)):
                #print(yezi.split('_')[1])
                if node[k].input[0] == node[yezi].output[0]:
                    if node[k].op_type in Next_OP_list: #if (4 in test_list):
                        list_yezi_ebm.append(k)
                        list_ebm.append(k)
                    if node[k].op_type in Stop_OP_list:
                        list_ebm.append(k)
                #elif node[k].op_type == 'Add' and node[k].input[1] == node[yezi].output[0]:
                #    list_ebm.append(k)
            #print(list_ebm)
            if len(list_ebm) == 1:
                if node[list_ebm[0]].op_type in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_node_ebm.append(node_tree)
                elif node[list_ebm[0]].op_type not in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_tree_node_ebm.append(node_tree)
            if len(list_ebm) == 2:
                if node[list_ebm[0]].op_type in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_node_ebm.append(node_tree)
                    
                elif node[list_ebm[0]].op_type not in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_tree_node_ebm.append(node_tree)
                    
                
                if node[list_ebm[1]].op_type in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_node_ebm.append(node_tree)
                    
                elif node[list_ebm[1]].op_type not in Next_OP_list:
                    node_tree = tree_node.insertleft(list_ebm[1])
                    list_tree_node_ebm.append(node_tree)
                    
        if len(list_yezi_ebm) != 0:
            #print(list_node_ebm)
            dsf(list_yezi_ebm, list_node_ebm)
    
    dsf(list_yezi, list_tree_node)

    #return node_0
    #print('先序遍历为:')
    save_tree_path = "save_tree"+"/"+str(prune_nodel_num)+'_Tree.gv'
    node_0.printtree(save_path=save_tree_path)
    data_list = []
    num_list = node_0.preorder(data_list)
    #print('-------node_0---------', node_0)
    return num_list, node_0
    #node_0.printtree()

def dsf_find_new(node, prune_nodel_num, Next_OP_list, Stop_OP_list):

    #name = str(prune_nodel_num) + '_node'
    #_node_0 = BinaryTree(prune_nodel_num)
    #rl = r.insertleft('b')
    #rr = r.insertright('c')
    #print('先序遍历为:')
    #r.preorder()


    #构建树的根节点，并赋值
    node_0 = BinaryTree()
    node_0.setrootvalue(prune_nodel_num)
    
    list_yezi = []
    list_yezi.append(prune_nodel_num)
    #print(list_yezi)
    list_tree_node = []
    list_tree_node.append(node_0)


    def dsf(list_yezi, list_tree_node):
        list_yezi_ebm = []
        list_tree_node_ebm = []
        for yezi,tree_node in zip(list_yezi, list_tree_node):
            list_ebm = []
            list_node_ebm = []
            for k in range(len(node)):
                #print(yezi.split('_')[1])
                if node[k].input[0] == node[yezi].output[0]:
                    if node[k].op_type in Next_OP_list: #if (4 in test_list):
                        list_yezi_ebm.append(k)
                        list_ebm.append(k)
                    if node[k].op_type in Stop_OP_list:
                        list_ebm.append(k)                
                if len(node[k].input)>1:
                    if node[k].input[1] == node[yezi].output[0]:
                        if node[k].op_type in Next_OP_list: #if (4 in test_list):
                            list_yezi_ebm.append(k)
                            list_ebm.append(k)
                        if node[k].op_type in Stop_OP_list:
                            list_ebm.append(k)

            if len(list_ebm) == 1:
                if node[list_ebm[0]].op_type in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_node_ebm.append(node_tree)
                elif node[list_ebm[0]].op_type in Stop_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_tree_node_ebm.append(node_tree)
            if len(list_ebm) == 2:
                if node[list_ebm[0]].op_type in Next_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_node_ebm.append(node_tree)
                    
                elif node[list_ebm[0]].op_type in Stop_OP_list:
                    node_tree = tree_node.insertright(list_ebm[0])
                    list_tree_node_ebm.append(node_tree)
        
                if node[list_ebm[1]].op_type in Next_OP_list:
                    node_tree = tree_node.insertleft(list_ebm[1])
                    list_node_ebm.append(node_tree)
                    
                elif node[list_ebm[1]].op_type in Stop_OP_list:
                    node_tree = tree_node.insertleft(list_ebm[1])
                    list_tree_node_ebm.append(node_tree)
                    
        if len(list_yezi_ebm) != 0:
            #print(list_node_ebm)
            dsf(list_yezi_ebm, list_node_ebm)
    
    dsf(list_yezi, list_tree_node)
    data_list = []
    num_list = node_0.preorder(data_list)
    
    save_tree_path = "save_tree"+"/"+str(prune_nodel_num)+'_Tree.gv'
    node_0.printtree(save_path=save_tree_path, nodee=node)
       
    return num_list, node_0

#查找二叉树的叶子节点
def find_leaf_nodes(root):
    if not root:
        return []

    if not root.leftchild and not root.rightchild:
        return [root.data]

    leaf_nodes = []
    if root.leftchild:
        leaf_nodes += find_leaf_nodes(root.leftchild)
    if root.rightchild:
        leaf_nodes += find_leaf_nodes(root.rightchild)

    return leaf_nodes

def init_note_rate(node, prune_node_list):
    for key in prune_node_list:
        if key == 'Conv':
            rate = prune_node_list[key]
            prune_node_list.clear()
            for i in range(len(node)):
                if(node[i].op_type=='Conv'):
                    prune_node_list[node[i].name] = rate
            break
    return prune_node_list

def find_prune_node_id(node, prune_node_list):
    prune_node_id_list = {}
    for key in prune_node_list.keys():#遍历所有需要剪枝的节点
        #根据key(节点名称)找到节点id
        prune_nodel_num = find_node_id(node, key)
        prune_node_id_list[prune_nodel_num] = prune_node_list[key]
    return prune_node_id_list