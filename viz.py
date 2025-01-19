import torch 
import numpy as np
import matplotlib.pyplot as plt

def get_patch_coordinates(patch_index, patch_size, image_shape):
    num_patches_per_row = image_shape[2] // patch_size
    row = patch_index // num_patches_per_row
    col = patch_index % num_patches_per_row
    return row * patch_size, col * patch_size

def visualize_selected_patches(image, model_agent, device):
    model_agent.eval()
    with torch.no_grad():
        patches = model_agent.select_action(image.to(device))
        patches = [1 if patch > patches.mean() else 0 for patch in patches]
    
    patch_size = model_agent.patch_size
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    
    for i, patch in enumerate(patches):
        if patch == 1:
            y, x = get_patch_coordinates(i, patch_size, image.shape)
            rect = plt.Rectangle((x, y), patch_size, patch_size, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    plt.show()

def visualize_attention_scores(image, model_vit, device):
    model_vit.eval()
    with torch.no_grad():
        attention_scores = model_vit.get_attention_scores(image.to(device))
    
    attention_map = attention_scores.mean(dim=1).cpu().numpy()
    attention_map = np.mean(attention_map, axis=0)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax.imshow(attention_map, cmap='jet', alpha=0.5)
    
    plt.show()