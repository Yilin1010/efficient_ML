#   Author: Yilin Tang
#   Date: 2024-04-24
#   CS 5330 Computer Vision
#   Description: visualize weight distribution

import matplotlib
# matplotlib.use('TkAgg')  # Set the backend to TkAgg for X11
import matplotlib.pyplot as plt
import numpy as np
import os
import model 


def plot_distribution(model,file_name,named_keyword=['attention','dense','projection'],saveON=False):
  print(f"plot {file_name}")
  params = []
  print(named_keyword)
  for name, param in model.named_parameters():
    if 'weight' in name:
        if named_keyword is None or any(keyword in name for keyword in named_keyword):
          params.extend(param.data.flatten().tolist())

  # Find the minimum and maximum param values
  min_param = min(params)
  max_param = max(params)

  max_abs_param = max(abs(min_param),abs(max_param))

  # Create balanced bins from -max_abs_weight to max_abs_weight
  num_bins = 100
  bin_edges = np.linspace(-max_abs_param, max_abs_param, num_bins + 1)

  # Bin the weights and get the bin counts
  bin_counts, _ = np.histogram(params, bins=bin_edges)

  # Plot the histogram with the bin counts and bin edges
  plt.figure(figsize=(10, 6))
  plt.bar(bin_edges[:-1], bin_counts, width=np.diff(bin_edges), color='#9999ff', log=True)


  plt.xlabel('Weight Value')
  plt.ylabel('Frequency')
  # plt.title('')

#   plt.yscale('log')
#   # plt.yscale('log', ylim=(1, 10**8))
  plt.savefig(f'result/figure/{file_name}.pdf', dpi=300, format='pdf', bbox_inches='tight')


def create_figure_with_grids(models, file_name):
    # Create a figure with a 1x5 grid of axes
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Model Weight Distributions")

    # Iterate over models and plot distribution on each subplot
    for ax, model in zip(axes.flatten(), models):
        plot_distribution(model, ax)

    # Adjust layout and save the entire figure
    plt.tight_layout()
    plt.savefig(f'result/figure/{file_name}.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

def plot_figures(model_dir,named_keyword=['attention','dense','projection'],saveON=True):
  model_paths = model.return_all_model_paths(model_dir)

  for model_path in model_paths:
    loaded_model = model.load_saved_transformer(model_path)
    file_name = model_path.replace("/","-").replace(".pth","")
    plot_distribution(loaded_model,file_name,named_keyword,saveON)
    
    
### Example usage
# plot_figures('model/pruned/structured-global-sensitivity')