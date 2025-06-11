import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime import InferenceSession
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import onnx
import io
#import netron
#import evaluate
import os
import onnxruntime
from data.dataset import MyData, MyData_one, MyData_imagenet
from ctypes import cdll

#cdll.LoadLibrary('/home/rdd/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8') # ok
#cdll.LoadLibrary('libcuda.so') # ng

onnx_model = onnx.load("pytorch_checkpoint_cifar10/VGG16/ckpt_cifar10_prune_l1_tree_0.9.onnx")
save_ft_model = "pytorch_checkpoint_cifar10/VGG16/ckpt_cifar10_finetune_l1_tree_0.9.onnx"
onnx_model.opset_import[0].version = 13

provider = "GPUExecutionProvider"
onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=[provider])

input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表
    
    input_info = {
        "name" : input_tensor.name,
        "type" : input_tensor.type,
        "shape": input_tensor.shape,
    }
    print(input_info)



output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表
    
    output_info = {
        "name" : output_tensor.name,
        "type" : output_tensor.type,
        "shape": output_tensor.shape,
    }
    print(output_info)


input_name = input_info['name']
output_name = output_info['name']
print(output_name)


class MNISTTrainingBlock(onnxblock.TrainingBlock):
    def __init__(self):
        super(MNISTTrainingBlock, self).__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name), output_name

# Build the onnx model with loss
training_block = MNISTTrainingBlock()
for param in onnx_model.graph.initializer:
    print(param.name)
    training_block.requires_grad(param.name, True)

# Building training graph and eval graph.
model_params = None
with onnxblock.base(onnx_model):
    _ = training_block(*[output.name for output in onnx_model.graph.output])
    training_model, eval_model = training_block.to_model_proto()
    model_params = training_block.parameters()

# Building the optimizer graph
optimizer_block = onnxblock.optim.AdamW()
with onnxblock.empty_base() as accessor:
    _ = optimizer_block(model_params)
    optimizer_model = optimizer_block.to_model_proto()

onnxblock.save_checkpoint(training_block.parameters(), "onnx_train_save/checkpoint.ckpt")
onnx.save(training_model, "onnx_train_save/training_model.onnx")
onnx.save(optimizer_model, "onnx_train_save/optimizer_model.onnx")
onnx.save(eval_model, "onnx_train_save/eval_model.onnx")


batch_size = 100
train_kwargs = {'batch_size': batch_size}
test_batch_size = 100
test_kwargs = {'batch_size': test_batch_size}
train_epoch = 10

#kwargs = {'num_workers': 1, 'pin_memory': True}
data_root = '/data/rdd/cifar10_dataset/'

'''
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
'''

train_data = MyData(data_root, train=True)
val_data = MyData(data_root, train=False)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
print('train dataset len: {}'.format(len(train_loader.dataset)))


test_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=False)
print('val dataset len: {}'.format(len(test_loader.dataset)))


'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

'''

# Create checkpoint state.
state = CheckpointState.load_checkpoint("onnx_train_save/checkpoint.ckpt")

# Create module.
#print('@@@@@@@@@')
model = Module("onnx_train_save/training_model.onnx", state, "onnx_train_save/eval_model.onnx", device = "cuda")
#print('@@@@@@@@@')
# Create optimizer.
optimizer = Optimizer("onnx_train_save/optimizer_model.onnx", model)


# Util function to convert logits to predictions.
def get_pred(logits):
    return np.argmax(logits, axis=1)

# Training Loop :
def train(epoch):
    model.train()
    losses = []
    lr = 0.001
    if epoch >4:
        lr = lr * 0.1
        #print(lr)
    if epoch >7:
        lr = lr * 0.01
        #print(lr)
    optimizer.set_learning_rate(lr)
    for _, (data, target) in enumerate(train_loader):
        #print(data.shape)
        forward_inputs = [data.numpy(),target.numpy().astype(np.int64)]
        train_loss, _ = model(*forward_inputs)
        optimizer.step()
        print('train:', epoch, 'lr:', optimizer.get_learning_rate(), 'loss:', train_loss)
        
        model.lazy_reset_grad()
        losses.append(train_loss)
    print('optimizer.get_learning_rate()', optimizer.get_learning_rate())
    print(f'Epoch: {epoch+1},Train Loss: {sum(losses)/len(losses):.4f}')

# Test Loop :
def test(epoch):
    model.eval()
    losses = []
    correct = 0
    total = 0
    #metric = evaluate.load('accuracy')
    print('test:')

    for _, (data, target) in enumerate(test_loader):
        print('-----------------------')
        forward_inputs = [data.numpy(),target.numpy().astype(np.int64)]
        test_loss, logits = model(*forward_inputs)
        #print('test:', epoch, ':', test_loss)
        
        for num_picture in range(test_batch_size):
            #print(num_picture)
            pred = np.argsort(logits[num_picture])[::-1][0]
            gt = target.numpy()[num_picture]
            if gt == pred:
                correct = correct + 1
            total = total + 1
        #print(np.argsort(logits[0])[::-1][0])
        #print(logits[0])
        #print(target.numpy())
        #print('------------------------------------------------------------------')
        #metric.add_batch(references=target, predictions=get_pred(logits))
        losses.append(test_loss)
        print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Test ACC: {correct/total:.4f}')
    #metrics = metric.compute()
    print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Test ACC: {correct/total:.4f}')
    
test(0)

for epoch in range(train_epoch):
    train(epoch)
    test(epoch)
    model.export_model_for_inferencing(save_ft_model,["output"])
    #onnx.save(model,save_ft_model)

    
