def test_cases(num):
    if num == 1:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "optimizer":"Adam",
            "num_epochs": 25,
            "learning_rate": 3e-4
                }
    
    elif num == 2:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "optimizer":"AdamW",
            "num_epochs": 25,
            "learning_rate": 3e-4
                }
    
    elif num == 3:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "optimizer":"Adam",
            "num_epochs": 40,
            "learning_rate": 1e-4
                }
    
    elif num == 4:
        data_location =  "./dataset/flickr8k/data_with_tense"
        config = {
            "image_path": "./dataset/flickr8k/Images",
            "all_caption_data_path": data_location + "/Flickr8k.token.txt", 
            "train_data_path": data_location + "/train.txt",
            "test_data_path": data_location + "/test.txt",
            "validation_data_path": data_location + "/validation.txt",
            "optimizer":"AdamW",
            "num_epochs": 40,
            "learning_rate": 1e-4
                }

    
    else: raise ValueError("Test case not defined!")

    return config