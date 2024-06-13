#   Author: Yilin Tang
#   Date: 2024-04-24
#   CS 5330 Computer Vision
#   Description: 
#   load dataloader from tiny imagenet_200 and imagenet_1k
#   map classloc to classid

import os
import random
import shutil
import torch
import json
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from torchvision import datasets, transforms
from datasets import load_dataset
from transformers import ViTImageProcessor
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using GPU


def create_split_samplers(dataset_length=50000, 
                    splits={'pruning': 0.05, 'training': 0.10, 'val': 0.10}, seed=42):
    set_seed(seed)  # Set seed for reproducibility
    indices = list(range(dataset_length))
    np.random.shuffle(indices)
    
    split1 = int(np.floor(splits['pruning'] * dataset_length))
    split2 = split1 + int(np.floor(splits['training'] * dataset_length))
    split3 = split2 + int(np.floor(splits['val'] * dataset_length))
    
    pruning_indices, training_indices, val_indices = indices[:split1], indices[split1:split2], indices[split2:split3]
    
    pruning_sampler = SubsetRandomSampler(pruning_indices)
    training_sampler = SubsetRandomSampler(training_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    print(f"created split samplers, pruning: {len(pruning_indices)} training: {len(training_indices)} val: {len(val_indices)}")
    return pruning_sampler, training_sampler, val_sampler


class ImageNetSubset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.class_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self._load_image_files()


    def _load_image_files(self):
        self.image_files = []
        self.labels = []
        for class_dir in self.class_dirs:
            full_class_path = os.path.join(self.root, class_dir)
            image_filenames = [f for f in os.listdir(full_class_path) if f.endswith('.JPEG')]
            for filename in image_filenames:
                full_image_path = os.path.join(full_class_path, filename)
                if self._validate_image(full_image_path):
                    self.image_files.append(os.path.join(class_dir, filename))                       
                    class_id = int(class_dir)
                    
                    self.labels.append(class_id)

                else:
                    # If the image is invalid and cannot be deleted, it should be logged or handled appropriately
                    try:
                        os.remove(full_image_path)
                        print(f"Removed corrupted image: {full_image_path}")
                    except OSError as e:
                        print(f"Error removing {full_image_path}: {e}")

    def _validate_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that this is an image
            return True
        except (IOError, OSError):
            return False

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.root, image_file)
        image = Image.open(image_path).convert('RGB')  # Conversion is safe after validation
        label = self.labels[index]

        if self.transform:
            image = self.transform(image,return_tensors='pt')

        return image['pixel_values'][0], label


def tiny_imagenet_collate_fn(model_name='google/vit-base-patch16-224',
                             data_info = 'data/tiny_infos.json'):
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Set up data transformations
    transform = transforms.Compose([
           transforms.Lambda(lambda x: x.convert('RGB'))
      ])

    synset_to_clsidx_map = get_synset_to_clsidx_map()

    with open(data_info,'r') as f:
        dataset_info = json.load(f)
        tiny_imagenet_info = dataset_info['Maysee--tiny-imagenet']
        synsets = tiny_imagenet_info['features']['label']['names']

        clsloc_to_clsidx = [synset_to_clsidx_map.get(synset,-1) for synset in synsets]      
            
    def collate_fn(batch):
        # images = [transform(item['image']) for item in batch]
        # map clslocs to classid
        # labels = [int(clsloc_to_clsidx[item['label']]) for item in batch]

        images,labels = [],[]
        for item in batch:
          class_index = clsloc_to_clsidx[item['label']]
          if class_index == -1:
              continue  # Skip this item if the label maps to -1
          # Process the image and append to list if the label is valid
          image = transform(item['image'])
          images.append(image)
          labels.append(class_index)
      
        processed_images = image_processor(images, return_tensors='pt') 
        labels = torch.tensor(labels)
        return processed_images['pixel_values'], labels


    return collate_fn


def create_imagenet_classset(val_dir='data/imagenet_val',
                            label_file='data/ILSVRC2012_validation_ground_truth.txt',
                            classset_dir='data/classset_val', val_amount = 50000):
    
    unlabel = -1

    # Create the classset directory if it doesn't exist
    classset_dir = os.path.expanduser(classset_dir)
    os.makedirs(classset_dir, exist_ok=True)

    clsloc_to_clsidx = create_clsloc_to_clsidx()

    # Read the validation ground truth labels
    with open(label_file, 'r') as f:
        lines = f.readlines()
        if len(lines) != val_amount:
            raise ValueError(f"label file should has {val_amount} lines")
        # Map clsloc to class ID using clsloc_to_clsidx
        mapped_class_labels = [int(clsloc_to_clsidx.get(line.strip(),unlabel)) for line in lines]
    
    print(f"map {len(mapped_class_labels)} elements")
    # Create a dictionary to store the image filenames for each class
    class_images = {}
    for class_label in set(mapped_class_labels):
          class_images[class_label] = []
          # Create a subdirectory for the class if it doesn't exist
          class_dir_val = os.path.join(classset_dir, str(class_label))
          os.makedirs(class_dir_val, exist_ok=True)

    print(f"load {len(class_images)} classes")

    # Iterate over the validation set images and add filenames to the corresponding class list
    for i, filename in enumerate(sorted(os.listdir(val_dir))):
        if i+1 > val_amount:
            print(filename)
            raise ValueError(f"val dir should have {val_amount} images, current is {i+1}") 
        # Get the class label for the image
        class_label = mapped_class_labels[i]
        class_images[class_label].append(filename)

    if unlabel in class_images:
        print(f"del {len(class_images[unlabel])} images label = {unlabel}")
        del class_images[unlabel]
        os.rmdir(os.path.join(classset_dir, str(unlabel)))

    
    # Move all images to class subset
    print(f"start moving ") 
    for class_id, filenames in class_images.items():
        if len(filenames) > 50:
            print(class_id,filename)
            print("each class shouldn't have more than 50 images for val set")
        for filename in filenames:
            # Copy the subset images to the corresponding class directory
            src_path = os.path.join(val_dir, filename)
            dst_path = os.path.join(classset_dir, str(class_id), filename)
            shutil.move(src_path, dst_path)
        if class_id%50==1:
            print(f"created {class_id} classset")



def create_imagenet_subset(val_dir='data/imagenet_val',
                           label_file='data/ILSVRC2012_validation_ground_truth.txt',
                           subset_dir_val='data/subset_val',
                           subset_dir_pruning ='data/subset_pruning',
                           num_samples_per_class_val=10,
                           num_samples_per_class_pruning=10):
    
    clsloc_to_clsidx = create_clsloc_to_clsidx()
    
    # Read the validation ground truth labels
    with open(label_file, 'r') as f:

        # Map clsloc to class ID using clsloc_to_clsidx
        mapped_class_labels = [int(clsloc_to_clsidx.get(line.strip(), -1)) for line in f]


    # Create the subset directory if it doesn't exist
    subset_dir_val = os.path.expanduser(subset_dir_val)  # Expands user directory correctly
    subset_dir_pruning = os.path.expanduser(subset_dir_pruning)
    os.makedirs(subset_dir_val, exist_ok=True)
    os.makedirs(subset_dir_pruning, exist_ok=True)

    # Create a dictionary to store the image filenames for each class
    class_images = {}
    for class_label in set(mapped_class_labels):
        class_images[class_label] = []

     # Iterate over the validation set images and add filenames to the corresponding class list
    for i, filename in enumerate(sorted(os.listdir(val_dir))):
        # Get the class label for the image
        class_label = mapped_class_labels[i]
        class_images[class_label].append(filename)

        # Create a subdirectory for the class if it doesn't exist
        class_dir_val = os.path.join(subset_dir_val, str(class_label))
        class_dir_pruning = os.path.join(subset_dir_pruning, str(class_label))
        os.makedirs(class_dir_val, exist_ok=True)
        os.makedirs(class_dir_pruning, exist_ok=True)

        
    # Create the pruning and val subset by random sampling a fixed number of images per class
    # pruning and val subset Do Not cotaine same samples
    for class_label, filenames in class_images.items():
        subset_filenames = random.sample(filenames, min(num_samples_per_class_val+num_samples_per_class_pruning, len(filenames)))
        
        count=0
        # Copy the subset images to the corresponding class directory
        for count,filename in enumerate(subset_filenames):
            src_path = os.path.join(val_dir, filename) 
            if count > num_samples_per_class_pruning:
                dst_path = os.path.join(subset_dir_val, str(class_label), filename)
            else:
                dst_path = os.path.join(subset_dir_pruning, str(class_label), filename)
                
            shutil.move(src_path, dst_path)
            

    print(f"ImageNet subset created with {num_samples_per_class_val} images per class for val subset")
    print(f"ImageNet subset created with {num_samples_per_class_pruning} images per class for pruning subset")



def get_clsloc_to_synset_map(file_path='data/map_clsloc.txt'):
    clsloc_to_synset = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            synset_id = parts[0]  # first column is synset_id
            folder_index = parts[1]  # Second column is the folder index
            clsloc_to_synset[folder_index] = synset_id
    return clsloc_to_synset

def get_synset_to_clsidx_map(file_path='data/synsets.txt'):
    synset_to_clsidx = {}
    with open(file_path, 'r') as file:
        for class_id, line in enumerate(file): # class id is index
            sysnet_id = line.strip() # only col is sysnet
            synset_to_clsidx[sysnet_id] = class_id
    return synset_to_clsidx

def create_clsloc_to_clsidx():
    clsloc_to_synset = get_clsloc_to_synset_map()
    synset_to_clsidx = get_synset_to_clsidx_map()
    clsloc_to_clsidx = {}
    # Iterate through the clsloc_labels dictionary
    for folder_index, label in clsloc_to_synset.items():
        if label in synset_to_clsidx:
            clsloc_to_clsidx[folder_index] = synset_to_clsidx[label]
    return clsloc_to_clsidx


def split_load_imagenet(classset_dir ='data/classset_val',
                    model_name='google/vit-base-patch16-224',
                    num_workers=2, train=False, val=False,prune=False):
    
    assert train or val or prune,"please set train, val or pruning to true"
    # Define your transformations if needed
    image_processor  = ViTImageProcessor.from_pretrained(model_name)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Create Dataset
    dataset = ImageNetSubset(root=classset_dir, transform=image_processor)

    print("start spliting dataset")
    # Define splits
    # splits = {'pruning': 5/50, 'training': 5/50, 'val': 5/50} 
    
    # debug
    splits = {'pruning': 10/50, 'training': 10/50, 'val': 10/50}

    # Create Samplers
    pruning_sampler, training_sampler, val_sampler = create_split_samplers(len(dataset), splits)

    dataloaders = []  
    if train:
        training_loader = DataLoader(dataset, batch_size=16, sampler=training_sampler, shuffle=False, num_workers=num_workers)
        print("return training set")
        dataloaders.append(training_loader)
    if val:
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler, shuffle=False, num_workers=num_workers)
        print("return val set")
        dataloaders.append(val_loader)
    if prune:
        pruning_loader = DataLoader(dataset, batch_size=16, sampler=pruning_sampler, shuffle=False, num_workers=num_workers)
        print("return pruning set")
        dataloaders.append(pruning_loader)
    
    if len(dataloaders)==1:return dataloaders[0]
    else:return tuple(dataloaders)


def load_imagenet_subset(subset_dir='data/subset_val',model_name='google/vit-base-patch16-224'):

    
    # Set up data transformations
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # Create a custom dataset for the validation subset
    

    subset_dataset = ImageNetSubset(root=subset_dir, transform=image_processor)
    subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=True)

    return subset_loader



def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader


def load_tiny_imagenet(model_name='google/vit-base-patch16-224',data_path = "Maysee/tiny-imagenet", 
                       data_info = 'data/tiny_infos.json',**kwargs):
    
    set_seed()

    train_batch_size = kwargs.get('train_batch',32)

    print("loading tiny")
    tiny_imagenet_train = load_dataset(data_path, split='train')
    tiny_imagenet_test = load_dataset(data_path, split='valid')


    collate_fn = tiny_imagenet_collate_fn(model_name, data_info)

    # Use DataLoader to handle batches and shuffling
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_train, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(tiny_imagenet_test, batch_size=32, collate_fn=collate_fn, shuffle=True)

    return train_loader, test_loader


