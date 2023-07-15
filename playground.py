"""

Check the feature map size from 

1. ResNet 18/50

2. ViT

"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

model = resnet18(pretrained=False)

# print("layer update check ")
# print(model.conv1)

model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
upconv1     = nn.ConvTranspose2d(64, 192, 3, stride=2, padding=1, dilation=1, output_padding=1)

# upconv2 = nn.ConvTranspose2d(192, 192, 3, stride=1, padding=1)
# exit()

model = nn.Sequential(*list(model.children())[:-5])
print("Model")
print(model)

input = torch.rand(1, 3, 64, 64)

out = model(input)
print("size after one model pass >>> ", out.shape)
out = upconv1(out)
# out = upconv2(out)

print("Output >> ")
print(out.shape)




