import numpy as np

def l2_tree_prune(prune_nodel_num, next_node_num, node, initializer_list, prune_rate, num_channet_sum):
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
            #L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
            l2_norm = np.sqrt(np.sum(wigth_np_copy*wigth_np_copy, axis=(1, 2, 3)))
            
            #print(l2_norm.shape)
            nood_norm = l2_norm

            #print(np.linalg.norm(X, axis=0))
            #weight_l1 = L1_norm
            #print(initializer_list[j].dims[0])
        if len(next_node_num)==1:
            if(initializer_list[j].name==node[next_node_num[0]].input[1]):
                if node[next_node_num[0]].op_type == "Conv":
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
                    #L1_norm_n = np.sum(wigth_np_copy_n, axis=(0, 2, 3))
                    l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0, 2, 3)))
                    nood_next_norm1 = l2_next_norm
                    #weight_next_l1 = L1_norm_n
                    #print(initializer_list[j].dims[0])
                if node[next_node_num[0]].op_type == "Gemm":
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
                    #L1_norm_n = np.sum(wigth_np_copy_n, axis=(0))
                    l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0)))
                    nood_next_norm1 = l2_next_norm
                    #weight_next_l1 = L1_norm_n
                    #print(initializer_list[j].dims[0])
        if len(next_node_num)==2:
            if(initializer_list[j].name==node[next_node_num[0]].input[1]):
                if node[next_node_num[0]].op_type == "Conv":
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
                    #L1_norm_n = np.sum(wigth_np_copy_n, axis=(0, 2, 3))
                    l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0, 2, 3)))
                    nood_next_norm1 = l2_next_norm
                    #weight_next_l1 = L1_norm_n
                    #print(initializer_list[j].dims[0])
                if node[next_node_num[0]].op_type == "Gemm":
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
                    #L1_norm_n = np.sum(wigth_np_copy_n, axis=(0))
                    l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0)))
                    nood_next_norm1 = l2_next_norm
                    #weight_next_l1 = L1_norm_n
                    #print(initializer_list[j].dims[0])
            if(initializer_list[j].name==node[next_node_num[1]].input[1]):
                if node[next_node_num[1]].op_type == "Conv":
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
                    #L1_norm_n = np.sum(wigth_np_copy_n, axis=(0, 2, 3))
                    l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0, 2, 3)))
                    nood_next_norm2 = l2_next_norm
                    #weight_next_l1 = L1_norm_n
                    #print(initializer_list[j].dims[0])
                if node[next_node_num[1]].op_type == "Gemm":
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
                    #L1_norm_n = np.sum(wigth_np_copy_n, axis=(0))
                    l2_next_norm = np.sqrt(np.sum(wigth_np_copy_n*wigth_np_copy_n, axis=(0)))
                    nood_next_norm2 = l2_next_norm
                    #weight_next_l1 = L1_norm_n
                    #print(initializer_list[j].dims[0])
    if len(next_node_num)==1:
        weight_Tree = nood_norm * nood_next_norm1[num_channet_sum : num_channet_sum + len(nood_norm)]
    if len(next_node_num)==2:
        weight_Tree = nood_norm * nood_next_norm1[num_channet_sum : num_channet_sum + len(nood_norm)] + nood_norm * nood_next_norm2[num_channet_sum : num_channet_sum + len(nood_norm)]

    
    for j in range(len(initializer_list)):
        if(initializer_list[j].name==node[prune_nodel_num].input[1]):
            
            arg_max = np.argsort(weight_Tree)
            np.savetxt('l2_tree'+str(prune_nodel_num)+'.txt', arg_max, fmt="%d")
            print('arg_max', arg_max)
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
    print(node[prune_nodel_num].name,':',prune_channel_num)

    if len(next_node_num)==1:
        for current_j in range(len(initializer_list)):
            #print(initializer_list[current_j].name)
            if(initializer_list[current_j].name==node[next_node_num[0]].input[1]):
                if node[next_node_num[0]].op_type == "Conv":
                    #print(node[current_num].input[1])
                    #print(initializer_list[j].name)
                    wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                    #print(initializer_list[current_j].dims)
                    wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                    #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
                    wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                    #重新赋值initializer
                    initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                    initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                if node[next_node_num[0]].op_type == "Gemm":
                    #print(node[current_num].input[1])
                    #print(initializer_list[j].name)
                    wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                    #print(initializer_list[current_j].dims)
                    wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                    #wigth_np = onnx.numpy_helper.to_array(initializer_list[current_j])
                    wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                    #重新赋值initializer
                    initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                    initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
    if len(next_node_num)==2:
        for current_j in range(len(initializer_list)):
            #print(initializer_list[current_j].name)
            if(initializer_list[current_j].name==node[next_node_num[0]].input[1]):
                if node[next_node_num[0]].op_type == "Conv":
                    wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                    wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                    wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                    initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                    initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                if node[next_node_num[0]].op_type == "Gemm":
                    wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                    wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                    wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                    initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                    initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
            if(initializer_list[current_j].name==node[next_node_num[1]].input[1]):
                if node[next_node_num[1]].op_type == "Conv":
                    wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                    wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                    wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                    initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                    initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                if node[next_node_num[1]].op_type == "Gemm":
                    wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                    wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                    wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                    initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                    initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]

    #print()