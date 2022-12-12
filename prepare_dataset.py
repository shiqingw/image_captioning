
if __name__ == "__main__":
    dataset_path = "./dataset/flickr8k"
    all_captions = dataset_path + "/captions.txt"
    train_imgs_file = dataset_path + "/Flickr_8k.trainImages.txt"
    validation_imgs_file = dataset_path + "/Flickr_8k.devImages.txt"
    test_imgs_file = dataset_path + "/Flickr_8k.testImages.txt"

    with open(all_captions, "r") as f:
        data = f.read()
    
    grouped_captions = {}
    for line in data.split("\n"):
        caption_data = line.split(".jpg,")
        if len(caption_data)<2:
            continue
        image_id, image_caption = caption_data[0], caption_data[1]
        if image_id not in grouped_captions:
            grouped_captions[image_id] = []

        image_caption = image_caption.replace('"', '')
        image_caption = image_caption.lower()
        grouped_captions[image_id].append(image_caption)
    
    with open(train_imgs_file, "r") as f:
        data = f.read()
    train_imgs = data.split("\n")

    with open(validation_imgs_file, "r") as f:
        data = f.read()
    validation_imgs = data.split("\n")

    with open(test_imgs_file, "r") as f:
        data = f.read()
    test_imgs = data.split("\n")

    def save_captions(image2caption, subset_imgs, save_path):
        captions = []
        for image_name in subset_imgs:
            image_id = image_name.split('.')[0]
            if image_id in image2caption:
                for caption in image2caption[image_id]:
                    captions.append("{} {}\n".format(image_name, caption))
        with open(save_path, "w") as f:
            f.writelines(captions)
    
    save_captions(grouped_captions, train_imgs, dataset_path + "/train.txt")
    save_captions(grouped_captions, test_imgs, dataset_path + "/test.txt")
    save_captions(grouped_captions, validation_imgs, dataset_path + "/validation.txt")

