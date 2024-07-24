#   Author: Yilin Tang
#   Date: 2024-04-24
#   CS 5330 Computer Vision
#   Description: 
#   apply structure and unstructure pruning,  
#   global pruning, do sensitivity analysis based on weight contribution for classification



import torch
from model.vit import load_pretrained_transformer,save_transformer,load_saved_transformer
from model.utils import return_all_model_paths
from dataset import load_imagenet,load_tiny_imagenet
from evaluate import get_evaluation
import os
from tqdm import tqdm
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def load_checkpoint(model,save_dir):
    # Check if a checkpoint exists in the finetune directory
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}. Starting from epoch {start_epoch+1}.')
    else:
        start_epoch = 0
        
    return model, start_epoch

def fine_tune(model, model_dir, data_loader=None, 
            learning_rate=1e-5, scheduler_type='cosine',sparse=False,save=True,checkpoint=True,**kwargs):
    
    epochs = kwargs.get('epochs',5)
    max_epoch = kwargs.get('max_epoch',10)
    if data_loader is None:
        data_loader = load_tiny_imagenet(train=True)
        
    save_dir = f'{model_dir}/finetune'
    
    model, start_epoch = load_checkpoint(model,save_dir) # start is finished epoch

    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    process = psutil.Process()

    # Define the learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    else:
        raise ValueError('Invalid scheduler type. Must be "cosine" or "plateau".')

    for epoch in range(start_epoch,epochs):
                
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix({'Loss': running_loss / (pbar.n + 1), 'Accuracy': 100 * correct_predictions / total_samples})

        # Update the learning rate scheduler
        if scheduler_type == 'cosine':
            scheduler.step()
        elif scheduler_type == 'plateau':
            scheduler.step(running_loss / len(data_loader))

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Get memory usage
        memory_usage = process.memory_info().rss / (1024 ** 2)  # in MB
        print(f'Training Memory Usage: {memory_usage:.2f} MB')
        
            
        if epoch + 1 in [1, 5, 8, 10] and save and checkpoint:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_accuracy
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
        if epoch +1 >= max_epoch or epoch_accuracy >= 80:
            break
            
    
    if sparse:
        model,save_dir = convert_to_sparse(model,model_dir)
    if save:
        model_path = save_transformer(model,save_dir,f'vit_epoch_{epoch+1}.pth')
        
    return model,model_path  # return fine-tuned model

def sensitivity_pruning_unstructured_global(model_name='google/vit-base-patch16-224', 
                        data_loader=None,sparse=False,
                        save=True,**kwargs):
            
    if data_loader == None:
        data_loader = load_imagenet(prune=True)
    num_batches = kwargs.get('num_batches', len(data_loader))  # Default
    pruning_ratio = kwargs.get('pruning_ratio', 0.5) 

    save_dir = f"model/pruned/unstructured-global-sensitivity/ratio{int(pruning_ratio*100)}%-batch{num_batches}"

    model = load_pretrained_transformer(model_name)
    model.to(device)
    model.eval()
    

    # Calculate the sensitivity of each weight
    sensitivity_scores = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            sensitivity_scores[name] = torch.zeros_like(param)

    
    pbar = tqdm(total=num_batches, desc="Calculating sensitivity scores", unit="batch")
    for _ in range(num_batches):
        # Get a random batch of data
        images, class_ids = next(iter(data_loader))
        class_ids = class_ids.to(device)
        images = images.to(device)
        images.requires_grad = True

        # Forward pass
        # Transformer expects the data as a dictionary of inputs.
        outputs = model(images)

        # Calculate gradients based on its impact on the final classification performance
        outputs.logits.sum().backward()

        # Accumulate sensitivity scores
        for name, param in model.named_parameters():
            if 'weight' in name:
                sensitivity_scores[name] += torch.abs(param.grad)

        pbar.update(1)

    pbar.close()

    # Normalize sensitivity scores
    for name in sensitivity_scores:
        sensitivity_scores[name] /= num_batches

    # Calculate the threshold for pruning
    total_elements = sum(scores.numel() for scores in sensitivity_scores.values())
    print(f"total elements of sensitivity scores: {total_elements}")
    threshold_index = int(total_elements * pruning_ratio)
    flattened_scores = torch.cat([scores.view(-1) for scores in sensitivity_scores.values()])
    threshold = flattened_scores.kthvalue(threshold_index).values.item()

    # Create a binary mask for each weight tensor
    masks = {}
    for name, scores in sensitivity_scores.items():
        masks[name] = scores >= threshold

    # Apply pruning using the binary masks
    pruned_model = apply_pruning(model,masks)

    # save pruned model pth 
    # Save the pruned model
    if sparse:
        model,save_dir= convert_to_sparse(model,save_dir,save)
    if save:
        model_path = save_transformer(model,save_dir)

    return model, model_path



def sensitivity_pruning_structured_global(model_name='google/vit-base-patch16-224', 
                                          data_loader=None, sparse=False,
                                          save=True,**kwargs):
        

    if data_loader == None:
        data_loader = load_imagenet(prune=True)
    num_batches = kwargs.get('num_batches', len(data_loader))  # Default
    pruning_ratio = kwargs.get('pruning_ratio', 0.5)
    pruned_layer_keywords = ['projection','attention','dense'] # skip Embedding and Normalization classifier.weight layer


    save_dir = f"model/pruned/structured-global-sensitivity/ratio{int(pruning_ratio*100)}%-batch{num_batches}"
    print(save_dir)

    model = load_pretrained_transformer(model_name)
    model.to(device)
    model.eval()

    # Calculate the sensitivity of each weight
    sensitivity_scores_per_row = {}
    for name, param in model.named_parameters():
        if 'weight' in name and any(keyword in name for keyword in pruned_layer_keywords):

            # Initialize the shape to align with the Aggregating sum for row(filters)
            sensitivity_scores_per_row[name] = torch.zeros(param.size(0), device=param.device)

    pbar = tqdm(total=num_batches, desc="Calculating sensitivity scores", unit="batch")
    for _ in range(num_batches):
        # Get a random batch of data
        images, class_ids = next(iter(data_loader))
        class_ids = class_ids.to(device)
        images = images.to(device)
        images.requires_grad = True

        # Forward pass
        # Transformer expects the data as a dictionary of inputs.
        outputs = model(images)

        # Calculate gradients based on its impact on the final classification performance
        outputs.logits.sum().backward()

        # Accumulate sensitivity scores
        for name, param in model.named_parameters():
            if 'weight' in name and any(keyword in name for keyword in pruned_layer_keywords):
                param_grad = torch.abs(param.grad)
                try:
                    if len(param.shape) == 4:  # Convolutional layer
                        sensitivity_scores_per_row[name] += param_grad.sum(dim=(1, 2, 3))
                    elif len(param.shape) == 2:  # Fully-connected layer
                        sensitivity_scores_per_row[name] += param_grad.sum(dim=1)
                    else:  # Handle other tensor shapes that first two dimensions likely correspond to the input and output dimensions
                        sensitivity_scores_per_row[name] += param_grad.sum(dim=tuple(range(1, len(param.shape))))
                # Debug
                except RuntimeError as e:
                    print(f"Weight tensor name: {name}, shape: {param.shape}")  # Print weight tensor name and shape
                    print(f"param_grad.shape {param_grad.shape}")  # Print error message for problematic tensor
                    print(f"Error for weight tensor {name}: {e}")  # Print error message for problematic tensor


        pbar.update(1)

    pbar.close()


    # Normalize the row-wise sensitivity scores
    for name in sensitivity_scores_per_row:
        sensitivity_scores_per_row[name] /= sensitivity_scores_per_row[name].sum(dim=tuple(range(1, len(sensitivity_scores_per_row[name].shape))), keepdim=True)

    # Calculate the threshold for pruning
    total_rows = sum(scores.shape[0] for scores in sensitivity_scores_per_row.values())
    threshold_index = int(total_rows * pruning_ratio)
    flattened_scores = torch.cat([scores for scores in sensitivity_scores_per_row.values()])
    threshold = flattened_scores.kthvalue(threshold_index).values.item()

    # Create a binary mask for row-wise sensitivity scores
    masks = {}
    for name, scores_per_row in sensitivity_scores_per_row.items():
        param = model.state_dict()[name]
        mask = scores_per_row >= threshold 
        if len(param.shape) == 4:  # Convolutional layer
            mask = scores_per_row.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # prune entire filters
        elif len(param.shape) == 2:  # Fully-connected layer
            mask =  scores_per_row.unsqueeze(1)
        else:
            # Handle other tensor shapes
            mask = scores_per_row.unsqueeze(1).expand_as(param)
        masks[name] = mask >= threshold
        
    #  Apply pruning
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and any(keyword in name for keyword in pruned_layer_keywords):
                param.data = param.data * masks[name].to(device)

    # Save the pruned model
    if sparse:
        model,save_dir= convert_to_sparse(model,save_dir,save)
    if save:
        model_path = save_transformer(model,save_dir)

    return model, model_path


def apply_pruning(pruned_model,masks):
     # Apply pruning using the binary masks
    print("applying pruning")
    with torch.no_grad():
        for name, param in pruned_model.named_parameters():
            if 'weight' in name:
                param.data = param.data * masks[name].to(device)
    return pruned_model


def convert_to_sparse(model,model_dir=None):
    # Iterate over model parameters
    save_dir = f'{model_dir}/sparse'
    print(f"### Sparsing {save_dir}")
    
    
    for name, param in model.named_parameters():
        if param.dim() > 1:
            mask = param != 0
            indices = mask.nonzero().t()
            values = param[mask]
            model.state_dict()[name] = torch.sparse_coo_tensor(
                indices, values, param.size()
            )

    return model,save_dir



def evaluate_pruning(pruned_model_dir='model/pruned',base_model_name='google/vit-base-patch16-224', 
                     repruneOn=False, finetuneOn = False, structured=False,unstructured=False,keywords=None,**kwargs):
    
       
    # get pruned model paths from directory
    load_dir = os.path.join(pruned_model_dir)
    pruned_model_paths = return_all_model_paths(load_dir)
    
    print("found", pruned_model_paths) # debug
    
 
    # debug
    train_loader,test_loader = load_tiny_imagenet()
    
    if not pruned_model_paths or repruneOn:
        # If no pruned models found or repruning is requested, prune a new model
        print(f"No pruned models found or repruning requested. Pruning a new model on {base_model_name}")
        # pruned_model,pruned_model_path = sensitivity_pruning_unstructured_global(data_loader=prune_loader)
        if structured:
            pruned_model,pruned_model_path = sensitivity_pruning_structured_global(data_loader=train_loader,**kwargs)
            pruned_model_paths.append(pruned_model_path) if pruned_model_path not in pruned_model_paths else None
        if unstructured:
            pruned_model,pruned_model_path = sensitivity_pruning_unstructured_global(data_loader=train_loader,**kwargs)
            pruned_model_paths.append(pruned_model_path) if pruned_model_path not in pruned_model_paths else None

    
    # Load the pruned model and evluate it
    # test_loader = load_imagenet(val=True) 
    for pruned_model_path in reversed(pruned_model_paths):
        
        if 'checkpoint' in pruned_model_path:
            continue
       
        print(f"Evaluating pruned model: {pruned_model_path}")
        pruned_model = load_saved_transformer(pruned_model_path)
        # Evaluate the pruned model
        get_evaluation(pruned_model,pruned_model_path,base_model_name,test_loader)

        if finetuneOn and 'finetune' not in pruned_model_path:
            # train_loader = load_imagenet(train=True)
            model_dir = os.path.dirname(pruned_model_path)
            finetuned_model,fine_tune_model_path = fine_tune(pruned_model,model_dir=model_dir,data_loader=train_loader,**kwargs)
            get_evaluation(finetuned_model,fine_tune_model_path,base_model_name,test_loader)
        

def evaluate_finetune(dir='model',imagenet=True):
    paths = return_all_model_paths(dir) 
    if imagenet: 
        test_imagnet1k = load_imagenet(val=True)
    else:
        _,test_loader = load_tiny_imagenet()
    for path in paths:    
        if '1'or '5' or '8' or '7' or '10' or 'vit' in path:
            model = load_saved_transformer(path)
            if imagenet:
                get_evaluation(model,model_name=path+' on imagenet1k',test_loader=test_imagnet1k)
            else:
                get_evaluation(model,model_name=path+' on tiny imagenet 200',test_loader=test_loader)
                
