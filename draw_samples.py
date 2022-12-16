#imports
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import platform
import os
import argparse
import time
from nltk.translate.bleu_score import corpus_bleu

#custom imports 
from data_loader import FlickrDataset, get_data_loader
from utils import save_image, format_time, save_dict, plot_loss, plot_bleu_scores
from models import *
from test_case import test_cases

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ResNet+LSTM')
    parser.add_argument('--exp_num', default=0, type=int, help='test case number')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    exp_num = args.exp_num
    config = test_cases(exp_num)
    result_dir = './results/exp_{:03d}'.format(exp_num)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    if platform.system() == 'Darwin':
        if not torch.backends.mps.is_available():
            device = 'cpu'
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        else:
            device = 'mps'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> device: ', device)

    # location of the training data 
    BATCH_SIZE = 100
    NUM_WORKER = 4

    # defining the transform to be applied
    transforms = T.Compose([
        T.Resize(226),                     
        T.CenterCrop(224),                 
        T.ToTensor(),                               
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    # testing the dataset class
    dataset =  FlickrDataset(
        root_dir = config["image_path"],
        caption_file = config["all_caption_data_path"],
        transform=transforms,
        freq_threshold=config["freq_threshold"],
        vocab=None
    )

    # vocab_size
    vocab_size = len(dataset.vocab)
    print('==> Vocab_size:', vocab_size)

    #Hyperparams
    embed_size=300
    vocab_size = len(dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512

    #init model
    model = EncoderDecoder(
        embed_size=300,
        vocab_size = len(dataset.vocab),
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        device = device
    )

    checkpoint = torch.load(os.path.join(result_dir, "bahdanau_attention_model_state.pth"))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    sample_list_path = "./dataset/flickr8k/Flickr_8k.testImages.txt"
    with open(sample_list_path, "r") as f:
        data = f.read()
    name_list = data.split("\n")
    for i in range(50):
        img_name = name_list[i]
        img_location = os.path.join(config["image_path"],img_name)
        img = Image.open(img_location).convert("RGB")
        image = transforms(img).unsqueeze(0)
        features = model.encoder(image.to(device))
        caps, alphas = model.decoder.generate_caption_one(features,vocab=dataset.vocab)
        caption = ' '.join(caps)
        save_image(image[0], os.path.join(result_dir, "sample_{:03d}".format(i)), caption=caption)  
