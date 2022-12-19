import torch
import torchvision.models as models
# from models import EncoderDecoder
from torchsummary import summary

resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
summary(resnet, input_size=(3, 256, 256))

# model = EncoderDecoder(
#         embed_size=300,
#         vocab_size = 8505,
#         attention_dim=256,
#         encoder_dim=2048,
#         decoder_dim=512,
#         device = 'cpu'
#     )
# summary(model, input_size=[(3, 224, 224), 10])