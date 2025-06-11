import numpy as np


def l2_src_prune(prune_nodel_num, next_node_num, node, initializer_list, prune_rate, num_channet_sum,
                 prune_nodel_num1 = None, next_node_num1 = None, prune_rate1 = None,
                 prune_nodel_num2 = None, next_node_num2 = None, prune_rate2 = None):
    if prune_nodel_num1 is None and prune_nodel_num2 is None:
        #weight_l1 = []
        #weight_next_l1 = []
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num].input[1]):
                # 解码获得权重参数
                wight = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)

                wigth_np = wight.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
                
                #对权重矩阵进行重要性评估
                wigth_np_copy = np.abs(wigth_np)
                #L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
                l2_norm = np.sqrt(np.sum(wigth_np_copy*wigth_np_copy, axis=(1, 2, 3)))
                
                nood_norm = l2_norm
                
        weight_src = nood_norm

        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num].input[1]):
                
                arg_max = np.argsort(weight_src)
                print('arg_max:', arg_max)
                np.savetxt('l2_src'+str(prune_nodel_num)+'.txt', arg_max, fmt="%d")
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

        if len(next_node_num)==1:
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

        if len(next_node_num)==3:
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
                
                if(initializer_list[current_j].name==node[next_node_num[2]].input[1]):
                    if node[next_node_num[2]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[2]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
        
        if len(next_node_num)==4:
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
                
                if(initializer_list[current_j].name==node[next_node_num[2]].input[1]):
                    if node[next_node_num[2]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[2]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                
                if(initializer_list[current_j].name==node[next_node_num[3]].input[1]):
                    if node[next_node_num[3]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[3]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
        
    if prune_nodel_num1 is not None and prune_nodel_num2 is None:
        print("prune_nodel_num1 is not None and prune_nodel_num2 is None")
        #第一个Add_tree评估
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num].input[1]):
                # 解码获得权重参数
                wight = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                
                wigth_np = wight.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
                
                #对权重矩阵进行重要性评估
                wigth_np_copy = np.abs(wigth_np)
                L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
                
                nood_norm_1 = L1_norm
                #weight_l1 = L1_norm
                #print(initializer_list[j].dims[0])
        
        #第二个Add_tree评估
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num1].input[1]):
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
                nood_norm_2 = L1_norm
                #weight_l1 = L1_norm
                #print(initializer_list[j].dims[0])
        weight_src1 = nood_norm_1
        weight_src2 = nood_norm_2

        #对第一个Add_tree的根节点和偏执进行剪裁
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num].input[1]):
                arg_max = np.argsort(weight_src1)
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
        
        #对第二个Add_tree的根节点和偏执进行剪裁
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num1].input[1]):
                
                arg_max = np.argsort(weight_src2)
                #arg_max_rev = arg_max[::-1]
                prune_channel_num1 = arg_max[0 : round(prune_rate * len(arg_max))]
                #对重要性较小的几个通道进行剪裁
                wigth_np_prune = np.delete(wigth_np, prune_channel_num1, axis=0)
                #print(prune_channel_num)
                #重新赋值initializer
                #print(initializer_list[j].float_data)
                initializer_list[j].raw_data = wigth_np_prune.tobytes()
                initializer_list[j].dims[0] = wigth_np_prune.shape[0]
                #print(initializer_list[j].dims[0])

            if len(node[prune_nodel_num1].input)>2: #判断该节点是否存在偏置,如果存在，把偏置也剪裁掉
                if(initializer_list[j].name==node[prune_nodel_num1].input[2]):
                    bias = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                    #bias = onnx.numpy_helper.to_array(initializer_list[j])

                    bias_prune =  np.delete(bias, prune_channel_num1)
                    #重新赋值initializer
                    initializer_list[j].raw_data = bias_prune.tobytes()
                    initializer_list[j].dims[0] = bias_prune.shape[0]
        

        if len(next_node_num)==1:
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

        if len(next_node_num)==3:
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
                
                if(initializer_list[current_j].name==node[next_node_num[2]].input[1]):
                    if node[next_node_num[2]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[2]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
        
        if len(next_node_num)==4:
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
                
                if(initializer_list[current_j].name==node[next_node_num[2]].input[1]):
                    if node[next_node_num[2]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[2]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                
                if(initializer_list[current_j].name==node[next_node_num[3]].input[1]):
                    if node[next_node_num[3]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[3]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
        
    if prune_nodel_num1 is not None and prune_nodel_num2 is not None:
        print("prune_nodel_num1 is not None and prune_nodel_num2 is not None")
        #第一个Add_tree评估
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num].input[1]):
                # 解码获得权重参数
                wight = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                
                wigth_np = wight.reshape(initializer_list[j].dims[0], initializer_list[j].dims[1], initializer_list[j].dims[2], initializer_list[j].dims[3])
                
                #对权重矩阵进行重要性评估
                wigth_np_copy = np.abs(wigth_np)
                L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
                
                nood_norm_1 = L1_norm
                #weight_l1 = L1_norm
                #print(initializer_list[j].dims[0])
        
        #第二个Add_tree评估
        for k in range(len(initializer_list)):
            if(initializer_list[k].name==node[prune_nodel_num1].input[1]):
                # 解码获得权重参数
                #print(initializer_list[j].name)
                wight = np.frombuffer(initializer_list[k].raw_data, dtype=np.float32)
                #print(wight)
                #print(onnx.numpy_helper.to_array(initializer_list[j]))
                wigth_np = wight.reshape(initializer_list[k].dims[0], initializer_list[k].dims[1], initializer_list[k].dims[2], initializer_list[k].dims[3])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
                #print(wigth_np.shape)
                #对权重矩阵进行重要性评估
                wigth_np_copy = np.abs(wigth_np)
                L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
                #l2_norm = np.sqrt(np.sum(wigth_np_copy*wigth_np_copy, axis=(1, 2, 3)))
                
                #print(l2_norm.shape)
                nood_norm_2 = L1_norm
                #weight_l1 = L1_norm
                #print(initializer_list[j].dims[0])
        
        #第三个Add_tree评估
        for m in range(len(initializer_list)):
            if(initializer_list[m].name==node[prune_nodel_num2].input[1]):
                # 解码获得权重参数
                #print(initializer_list[j].name)
                wight = np.frombuffer(initializer_list[m].raw_data, dtype=np.float32)
                #print(wight)
                #print(onnx.numpy_helper.to_array(initializer_list[j]))
                wigth_np = wight.reshape(initializer_list[m].dims[0], initializer_list[m].dims[1], initializer_list[m].dims[2], initializer_list[m].dims[3])
                #wigth_np = onnx.numpy_helper.to_array(initializer_list[j])
                #print(wigth_np.shape)
                #对权重矩阵进行重要性评估
                wigth_np_copy = np.abs(wigth_np)
                L1_norm = np.sum(wigth_np_copy, axis=(1, 2, 3))
                #l2_norm = np.sqrt(np.sum(wigth_np_copy*wigth_np_copy, axis=(1, 2, 3)))
                
                #print(l2_norm.shape)
                nood_norm_3 = L1_norm
                #weight_l1 = L1_norm
                #print(initializer_list[j].dims[0])
        weight_src1 = nood_norm_1
        weight_src2 = nood_norm_2
        weight_src3 = nood_norm_3

        #对第一个Add_tree的根节点和偏执进行剪裁
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num].input[1]):
                arg_max = np.argsort(weight_src1)
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
        
        #对第二个Add_tree的根节点和偏执进行剪裁
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num1].input[1]):
                
                arg_max = np.argsort(weight_src2)
                #arg_max_rev = arg_max[::-1]
                prune_channel_num1 = arg_max[0 : round(prune_rate1 * len(arg_max))]
                #对重要性较小的几个通道进行剪裁
                wigth_np_prune = np.delete(wigth_np, prune_channel_num1, axis=0)
                #print(prune_channel_num)
                #重新赋值initializer
                #print(initializer_list[j].float_data)
                initializer_list[j].raw_data = wigth_np_prune.tobytes()
                initializer_list[j].dims[0] = wigth_np_prune.shape[0]
                #print(initializer_list[j].dims[0])

            if len(node[prune_nodel_num1].input)>2: #判断该节点是否存在偏置,如果存在，把偏置也剪裁掉
                if(initializer_list[j].name==node[prune_nodel_num1].input[2]):
                    bias = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                    #bias = onnx.numpy_helper.to_array(initializer_list[j])

                    bias_prune =  np.delete(bias, prune_channel_num1)
                    #重新赋值initializer
                    initializer_list[j].raw_data = bias_prune.tobytes()
                    initializer_list[j].dims[0] = bias_prune.shape[0]
        
        #对第三个Add_tree的根节点和偏执进行剪裁
        for j in range(len(initializer_list)):
            if(initializer_list[j].name==node[prune_nodel_num2].input[1]):
                
                arg_max = np.argsort(weight_src3)
                #arg_max_rev = arg_max[::-1]
                prune_channel_num2 = arg_max[0 : round(prune_rate2 * len(arg_max))]
                #对重要性较小的几个通道进行剪裁
                wigth_np_prune = np.delete(wigth_np, prune_channel_num2, axis=0)
                #print(prune_channel_num)
                #重新赋值initializer
                #print(initializer_list[j].float_data)
                initializer_list[j].raw_data = wigth_np_prune.tobytes()
                initializer_list[j].dims[0] = wigth_np_prune.shape[0]
                #print(initializer_list[j].dims[0])

            if len(node[prune_nodel_num2].input)>2: #判断该节点是否存在偏置,如果存在，把偏置也剪裁掉
                if(initializer_list[j].name==node[prune_nodel_num2].input[2]):
                    bias = np.frombuffer(initializer_list[j].raw_data, dtype=np.float32)
                    #bias = onnx.numpy_helper.to_array(initializer_list[j])

                    bias_prune =  np.delete(bias, prune_channel_num2)
                    #重新赋值initializer
                    initializer_list[j].raw_data = bias_prune.tobytes()
                    initializer_list[j].dims[0] = bias_prune.shape[0]
        
        
        if len(next_node_num)==1:
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

        if len(next_node_num)==3:
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
                
                if(initializer_list[current_j].name==node[next_node_num[2]].input[1]):
                    if node[next_node_num[2]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[2]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
        
        if len(next_node_num)==4:
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
                
                if(initializer_list[current_j].name==node[next_node_num[2]].input[1]):
                    if node[next_node_num[2]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[2]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                
                if(initializer_list[current_j].name==node[next_node_num[3]].input[1]):
                    if node[next_node_num[3]].op_type == "Conv":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1], initializer_list[current_j].dims[2], initializer_list[current_j].dims[3])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
                    if node[next_node_num[3]].op_type == "Gemm":
                        wight = np.frombuffer(initializer_list[current_j].raw_data, dtype=np.float32)
                        wigth_np = wight.reshape(initializer_list[current_j].dims[0], initializer_list[current_j].dims[1])
                        wigth_np_prune = np.delete(wigth_np, prune_channel_num + num_channet_sum, axis=1)
                        initializer_list[current_j].raw_data = wigth_np_prune.tobytes()
                        initializer_list[current_j].dims[1] = wigth_np_prune.shape[1]
        
    #print()