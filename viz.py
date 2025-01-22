import torch 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def get_patch_coordinates(patch_index, patch_size, num_patches_per_row):
    row = patch_index // num_patches_per_row
    col = patch_index % num_patches_per_row
    return row * patch_size , col * patch_size

def visualize_selected_patches(image, input_img, model_agent, device):
    # model_agent.eval()
    with torch.no_grad():
        patches = model_agent.select_action(input_img.to(device))
        patches = [1 if patch > patches.mean() else 0 for patch in patches]
    
    patch_size = model_agent.patch_size
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
   
    # ax.imshow(image.cpu().numpy())
    num_patches_per_row = int(image.size(1) // patch_size)
    for i, patch in enumerate(patches):
        if patch == 0:
            y, x = get_patch_coordinates(i, patch_size, num_patches_per_row)
            rect = plt.Rectangle((x-1, y-1), patch_size, patch_size, edgecolor='none', facecolor='gray', alpha=0.9)
            # rect = plt.Rectangle((x, y), patch_size, patch_size, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    plt.show()

def get_image_with_selected_patches(image, input_img, model_agent, device):
    with torch.no_grad():
        patches = model_agent.select_action(input_img.to(device))
        patches = [1 if patch > patches.mean() else 0 for patch in patches]
    
    patch_size = model_agent.patch_size
    num_patches_per_row = int(image.size(1) // patch_size)
    
    image_np = image.permute(1, 2, 0).cpu().numpy()
    for i, patch in enumerate(patches):
        if patch == 0:
            y, x = get_patch_coordinates(i, patch_size, num_patches_per_row)
            # Ensure y, x, and patch_size are integers
            print(f'y: {y}, type: {type(y)}')
            print(f'x: {x}, type: {type(x)}')
            print(f'patch_size: {patch_size}, type: {type(patch_size)}')
            image_np[y:y+patch_size, x:x+patch_size] = np.array([0.5, 0.5, 0.5])  # Gray color
    
    return image_np