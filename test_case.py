def test_cases(num):
    if num == 1:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 5,
            "optimizer":"Adam",
            "num_epochs": 25,
            "learning_rate": 3e-4,
            "weight_decay": 0,
            "scheduler": "None"
                }
    
    elif num == 2:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 5,
            "optimizer":"AdamW",
            "num_epochs": 25,
            "learning_rate": 3e-4,
            "weight_decay": 0,
            "scheduler": "None"
                }
    
    elif num == 3:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 5,
            "optimizer":"Adam",
            "num_epochs": 40,
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "scheduler": "None"
                }
    
    elif num == 4:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 5,
            "optimizer":"AdamW",
            "num_epochs": 40,
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "scheduler": "None"
                }
    
    elif num == 5:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 1,
            "optimizer":"Adam",
            "num_epochs": 25,
            "learning_rate": 3e-4,
            "weight_decay": 0,
            "scheduler": "None"
                }

    elif num == 6:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 1,
            "optimizer":"Adam",
            "num_epochs": 25,
            "learning_rate": 3e-4,
            "weight_decay": 0,
            "scheduler": "CosineAnnealingLR"
                }

    elif num == 7:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 1,
            "optimizer":"Adam",
            "num_epochs": 25,
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "scheduler": "None"
                }
    
    elif num == 8:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "freq_threshold": 1,
            "optimizer":"Adam",
            "num_epochs": 25,
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "scheduler": "CosineAnnealingLR"
                }

    
    else: raise ValueError("Test case not defined!")

    return config