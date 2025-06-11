import os, sys
 
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
 
 
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
 
 
img = cv2.imread("/mnt/d/dataset/ILSVRC2012_small/val/ILSVRC2012_val_00000001.JPEG")
 
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
 
to_tensor = transforms.ToTensor()
img = to_tensor(img)
img = img.unsqueeze_(0)
 
onnx_model_path = 'pytorch_checkpoint/resnet/resnet18_prune_l1_0.5.onnx'
rnet_session = onnxruntime.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'])
 
# compute ONNX Runtime output prediction
inputs = {rnet_session.get_inputs()[0].name: to_numpy(img)}
outs = rnet_session.run(None, inputs) # 推理
 
print(outs)