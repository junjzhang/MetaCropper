from model import Mars
import torch
input = torch.rand(1, 3, 256, 256)
ratio = torch.tensor([1.5])

net = Mars(net_type='vgg16', name_layer='15')
net(input, ratio)
