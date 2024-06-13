#   Author: Yilin Tang
#   Date: 2024-04-24
#   CS 5330 Computer Vision
#   Description: 
#   take model parameters and get evaluation

import os
import sys
import torch
from dataset import create_imagenet_subset
from evaluate import get_baseline_evaluation,get_evaluation
assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please select GPU"

def main():
  subset_dir = 'data/classset_val'
  if len(sys.argv) > 1:
    num_samples_per_class = int(sys.argv[1])
    subset_dir = f'{subset_dir}_{num_samples_per_class}'
  
  if not os.path.isdir(subset_dir):
      print(f"create {subset_dir}\n")
      create_imagenet_subset()


  if sys.argv[2]=="base":
    modelname = "google/vit-base-patch16-224"
  elif sys.argv[2]=="large":
    modelname = "google/vit-large-patch16-224"
  else:
      print("please select base or large transformer\n")
      return 
  
  print(f"Load {modelname} Evaluation Set: {subset_dir}")
  get_baseline_evaluation(modelname,subset_dir)


if __name__ == '__main__':
  main()

