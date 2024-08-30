import os


# return all pth model paths 
def return_all_model_paths(load_dir):
    model_paths = []
    if not os.path.exists(load_dir):
      print("No pth file found")
      return model_paths

  # Check if there are any saved pruned models
    for root, _, files in os.walk(load_dir):
        for file in files:
            if file.endswith(".pth"):  # Check for PyTorch model files
                # root is model dir containe config
                model_paths.append(os.path.join(root,file))

    return model_paths