import torch
import torchvision.models as models
from torchsummary import summary

resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
summary(resnet, input_size=(3, 224, 224))
