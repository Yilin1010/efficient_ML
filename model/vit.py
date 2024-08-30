#   Author: Yilin Tang
#   Date: 2024-04-24
#   CS 5330 Computer Vision
#   Description: load and save trained vision transformer


from transformers import AutoConfig, AutoModelForImageClassification
import torch
import os


def load_pretrained_transformer(model_name='google/vit-base-patch16-224'):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return model


def load_saved_transformer(model_path,config_dir='model'):
    
    # model_dir = os.path.dirname(model_path)
      
    # Define the device to use based on CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    # Load the configuration using AutoConfig
    config = AutoConfig.from_pretrained(config_dir)
    
    # Create the model using the loaded configuration
    model = AutoModelForImageClassification.from_config(config)
    
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # If the loaded object is a full checkpoint dictionary, use the 'model_state_dict' key
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Move the model to the defined device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    print(f'{model_path} loaded')
    
    return model
    

def save_transformer(model, save_dir, model_name='vit.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_path)
    # Optionally save the configuration file to easily reload the model architecture
    model.config.save_pretrained(save_dir)
    print(f'{model_path} saved')
    return model_path


