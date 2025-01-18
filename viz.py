import torch 
import numpy as np
import matplotlib.pyplot as plt

def visualize_selected_patches(image, model_agent, device):
    model_agent.eval()
    with torch.no_grad():
        patches = model_agent.select_patches(image.to(device))
    
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    
    for patch in patches:
        rect = plt.Rectangle((patch[1], patch[0]), patch[3] - patch[1], patch[2] - patch[0], edgecolor='r', facecolor='none')
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