#   Author: Yilin Tang
#   Date: 2024-04-24
#   CS 5330 Computer Vision
#   Description: 
#   evaluate flops, macs , size , memory usage and accuracy inference time for model

import torch
import time
from torchprofile import profile_macs
# from thop import profile
from vit import load_pretrained_transformer,return_all_model_paths
from dataset import load_imagenet,load_tiny_imagenet

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
usingcuda = torch.cuda.is_available()
print(f"using {device}")

def evaluate_model_performance(model, test_loader):
    
    model = model.to(device)
    model.eval()

    # Initialize metrics
    correct = 0
    total = 0
    total_time = 0

    if usingcuda:
        torch.cuda.reset_peak_memory_stats()  # Reset memory usage counter only if CUDA is available

    # Measure macs
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    macs = profile_macs(model, dummy_input)
    
    # Measure FLOPs 
    # Pass
    
    # Time and accuracy tracking 
    if usingcuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        
        for images, class_ids in test_loader:
            images = images.to(device)
            class_ids = class_ids.to(device)

            if usingcuda:
                start_event.record()
            else:
                start_time = time.time()

            outputs = model(images)

            if usingcuda:
                end_event.record()

            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == class_ids).sum().item()
            total += class_ids.size(0)

            if usingcuda:
                torch.cuda.synchronize()  # Wait for events to finish
                elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            else:
                elapsed_time = time.time() - start_time

            total_time += elapsed_time

    print(f"correct/total: {correct}/{total}")
   
    accuracy = correct / total
    inference_time = total_time / len(test_loader)
    throughput = total / total_time
    memory_usage = torch.cuda.max_memory_allocated() if usingcuda else 0  # Memory measurement only if CUDA is available

    return accuracy, macs, inference_time, throughput, memory_usage


def get_model_size(model):
    # model_size = sum(p.numel() for p in model.parameters()) * model.element_size()
    return sum(p.numel() * p.element_size() for p in model.parameters())



def get_baseline_evaluation(model_name:str='google/vit-base-patch16-224',
                   data_dir = 'data/classset_val',imagenet=False,tiny=True):
    
    print(f"loading model:{model_name}")
    model = load_pretrained_transformer(model_name)


    # model.element_size() gives the size in bytes per element
    model_size = get_model_size(model)

    if imagenet:
        print(f'Loading data:{data_dir}')
        test_loader = load_imagenet(data_dir,model_name,val=True)
        accuracy, macs, inference_time, throughput, memory = evaluate_model_performance(model, test_loader)
        print_save_evaluation(model_name+'on imagenet', accuracy, model_size, macs, inference_time, throughput, memory, "results.md")

    if tiny:
        _,test_loader = load_tiny_imagenet()
        accuracy, macs, inference_time, throughput, memory = evaluate_model_performance(model, test_loader)

        # Printing and saving evaluation results to md file
        print_save_evaluation(model_name+' on tiny imagenet', accuracy, model_size, macs, inference_time, throughput, memory, "results.md")


def get_evaluation(model, model_name: str, base_model_name:str='google/vit-base-patch16-224', test_loader=None, data_dir = 'data/classset_val'):
    
    if test_loader is None:
        print(f'Loading {data_dir}')
        test_loader = load_imagenet(data_dir, base_model_name,val=True)

    # model.element_size() gives the size in bytes per element
    model_size = get_model_size(model)

    accuracy, macs, inference_time, throughput, memory = evaluate_model_performance(model, test_loader)

    # Printing and saving evaluation results
    print_save_evaluation(model_name, accuracy, model_size, macs, inference_time, throughput, memory, "results.md")





def print_save_evaluation(model_name, accuracy, model_size, macs, inference_time, throughput, memory, file_path):
    
    results = (
        f"\n### Evaluation Results for {model_name}\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Model Size: {model_size / 1e6:.2f} MB\n"
        f"Macs: {macs / 1e9:.2f} G\n"  # G stands for Giga
        f"Inference Time: {inference_time:.2f} seconds per sample\n"
        f"Throughput: {throughput:.2f} samples/second\n"
        f"Memory Usage: {memory / 1e6:.2f} MB\n\n"  # Assuming memory is in bytes, converting to Megabytes for readability
    )
    print(results)
    with open(file_path, 'a') as f:
        f.write(results)



    