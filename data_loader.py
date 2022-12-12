#imports 
import os
from collections import Counter
import numpy as np
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from PIL import Image

class Vocabulary:
    #tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
        
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in Vocabulary.spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        
        #staring index 4
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]    
    
class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,caption_file,transform=None,freq_threshold=5,vocab=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(caption_file, "r") as f:
            self.data = f.read()

        self.imgs = []
        self.captions = []
        for sample in self.data.split("\n"):
            tokens = sample.split()
            if len(sample) < 2:
                # Image has no description: Invalid data row
                continue
            # First token is image id, remaining ones correspond to the caption
            image_name, image_caption = tokens[0], tokens[1:]
            image_id = image_name.split(".")[0] + '.jpg'
            # Recreate the description
            image_caption = " ".join(image_caption)
            self.imgs.append(image_id)
            self.captions.append(image_caption)
        
        if vocab == None:
            #Initialize vocabulary and build vocab
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions)
        else: 
            self.vocab = vocab
        
        self.inference_captions = self.group_captions(self.data)
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self,idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)

    def group_captions(self, data):
        """Groups captions which correspond to the same image.

        Main usage: Calculating BLEU score

        Arguments:
            data (list of str): Each element contains image name and corresponding caption
        Returns:
            grouped_captions (dict): Key - image name, Value - list of captions associated
                with that picture
        """
        grouped_captions = {}

        for line in data:
            caption_data = line.split()
            if len(caption_data)<2:
                continue
            image_name, image_caption = caption_data[0], caption_data[1:]
            image_id = image_name.split(".")[0] + '.jpg'
            if image_id not in grouped_captions:
                # We came across the first caption for this particular image
                grouped_captions[image_id] = []

            grouped_captions[image_id].append(image_caption)

        return grouped_captions
    
    def load_and_prepare_image(self, image_name):
        """Performs image preprocessing.

        Images need to be prepared for the ResNet encoder.
        Arguments:
            image_name (str): Name of the image file located in the subset directory
        """
        image_path = os.path.join(self.root_dir, image_name)
        img_pil = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image_tensor = self.transform(img_pil)
        return image_tensor

    def inference_batch(self, batch_size):
        """Creates a mini batch dataloader for inference.

        During inference we generate caption from scratch and in each iteration
        we feed words generated previously by the model (i.e. no teacher forcing).
        We only need input image as well as the target caption.
        """
        caption_data_items = list(self.inference_captions.items())

        num_batches = len(caption_data_items) // batch_size
        for idx in range(num_batches):
            caption_samples = caption_data_items[idx * batch_size: (idx + 1) * batch_size]
            batch_imgs = []
            batch_captions = []

            # Increase index for the next batch
            idx += batch_size

            # Create a mini batch data
            for image_name, captions in caption_samples:
                batch_captions.append(captions)
                batch_imgs.append(self.load_and_prepare_image(image_name))

            # Batch image tensors
            batch_imgs = torch.stack(batch_imgs, dim=0)
            if batch_size == 1:
                batch_imgs = batch_imgs.unsqueeze(0)

            yield batch_imgs, batch_captions

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets


def get_data_loader(dataset,batch_size,shuffle=False,num_workers=1):
    """
    Returns torch dataloader for the flicker8k dataset
    
    Parameters
    -----------
    dataset: FlickrDataset
        custom torchdataset named FlickrDataset 
    batch_size: int
        number of data to load in a particular batch
    shuffle: boolean,optional;
        should shuffle the datasests (default is False)
    num_workers: int,optional
        numbers of workers to run (default is 1)  
    """

    pad_idx = dataset.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx,batch_first=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader
