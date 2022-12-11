#imports
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import platform
import os
import argparse
import time
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

#custom imports 
from data_loader import FlickrDataset, get_data_loader
from utils import save_image, format_time, save_dict, plot_loss
from models import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ResNet+LSTM')
    parser.add_argument('--exp_num', default=0, type=int, help='test case number')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    exp_num = args.exp_num
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
    data_location =  "./dataset/flickr8k"
    BATCH_SIZE = 100
    # BATCH_SIZE = 6
    NUM_WORKER = 4

    # defining the transform to be applied
    transforms = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    # testing the dataset class
    dataset =  FlickrDataset(
        root_dir = data_location+"/Images",
        caption_file = data_location+"/Flickr8k.lemma.token.txt",
        transform=transforms
    )

    test_dataset =  FlickrDataset(
        root_dir = data_location+"/Images",
        caption_file = data_location+"/test.txt",
        transform=transforms,
        vocab=dataset.vocab
    )


    train_dataset =  FlickrDataset(
        root_dir = data_location+"/Images",
        caption_file = data_location+"/train.txt",
        transform=transforms,
        vocab=dataset.vocab
    )

    validation_dataset =  FlickrDataset(
        root_dir = data_location+"/Images",
        caption_file = data_location+"/validation.txt",
        transform=transforms,
        vocab=dataset.vocab
    )
    
    train_data_loader = get_data_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True
        )
    
    test_data_loader = get_data_loader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=False
        )
    
    validation_data_loader = get_data_loader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=False
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
    learning_rate = 3e-4

    #init model
    model = EncoderDecoder(
        embed_size=300,
        vocab_size = len(dataset.vocab),
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        device = device
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 25
    train_loss = []
    validation_loss = []
    test_loss = []
    best_loss = np.inf

    def train(epoch, train_loss): 
        model.train()
        epoch_train_loss = 0
        epoch_start_time = time.time()
        for batch_idx, (image, captions) in enumerate(iter(train_data_loader)):
            image, captions = image.to(device), captions.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            outputs, attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()
            epoch_train_loss += loss.item()
            
        epoch_end_time = time.time()
        print("Epoch: {:03d} | Loss: {:.3f} | Training time: {}".format(epoch,
         epoch_train_loss/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        train_loss += [epoch_train_loss/(batch_idx+1)]

    def validate(epoch, validation_loss): 
        model.eval()
        epoch_validation_loss = 0
        epoch_start_time = time.time()
        for batch_idx, (image, captions) in enumerate(iter(validation_data_loader)):
            image, captions = image.to(device), captions.to(device)
            outputs, attentions = model(image, captions)
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            epoch_validation_loss += loss.item()
            
        epoch_end_time = time.time()
        print("Epoch: {:03d} | Loss: {:.3f} | Validation time: {}".format(epoch,
         epoch_validation_loss/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        validation_loss += [epoch_validation_loss/(batch_idx+1)]

    def test(epoch, test_loss): 
        model.eval()
        epoch_test_loss = 0
        epoch_start_time = time.time()
        for batch_idx, (image, captions) in enumerate(iter(test_data_loader)):
            image, captions = image.to(device), captions.to(device)
            outputs, attentions = model(image, captions)
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            epoch_test_loss += loss.item()

        epoch_end_time = time.time()
        print("Epoch: {:03d} | Loss: {:.3f} | Testing time: {}".format(epoch,
         epoch_test_loss/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        test_loss += [epoch_test_loss/(batch_idx+1)]

        if epoch_test_loss/(batch_idx+1) < best_loss:
            best_loss = epoch_test_loss/(batch_idx+1)
            model_state = {
                    'num_epochs':num_epochs,
                    'embed_size':embed_size,
                    'vocab_size':len(dataset.vocab),
                    'attention_dim':attention_dim,
                    'encoder_dim':encoder_dim,
                    'decoder_dim':decoder_dim,
                    'state_dict':model.state_dict()
                }
            torch.save(model_state, os.path.join(result_dir, "bahdanau_attention_model_state.pth"))

    start_time = time.time()
    for epoch in range(num_epochs):
        train(epoch, train_loss)
        validate(epoch, validation_loss)
        test(epoch, test_loss)

        #generate the caption
        model.eval()
        with torch.no_grad():
            dataiter = iter(train_data_loader)
            img, _ = next(dataiter)
            image = img[0:1].to(device)
            features = model.encoder(image)
            caps, alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
            caption = ' '.join(caps)
            save_image(img[0], os.path.join(result_dir, "epoch_{:03d}".format(epoch)), title=caption)     
        
    stop_time = time.time()
    print("Total Time: %s" % format_time(stop_time - start_time))

    print("==> Saving training/validation/testing loss...")
    training_info = {"training_loss": train_loss, "validation_loss": validation_loss,
     "testing_loss": test_loss}
    save_dict(training_info, os.path.join(result_dir, "training_info.npy"))

    print("==> Drawing loss and acc...")
    loss_path = os.path.join(result_dir, "loss.png")
    plot_loss(train_loss, validation_loss, test_loss, loss_path)
    
    print("==> Process finished.")


