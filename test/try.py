from torch._C import dtype
import model
import torch
from loss import CropCrossEntropy
cri = CropCrossEntropy()
input = torch.rand(1, 3, 256, 256)
ratio = torch.tensor([[1.5, 0.7]])
acutal = torch.IntTensor(
    [[[100, 100, 200, 200], [50, 50, 100, 100]]])
net = model.Mars(net_type='vgg16', name_layer='15')
output = net(input, ratio)
loss = cri(output, acutal)
print(loss)
