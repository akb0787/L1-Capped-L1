# vgg16_pruning/prune.py
import torch
import torch.nn as nn
from .model import VGG16

def compute_l1_norm(filters):
    return torch.abs(filters).sum(dim=(1, 2, 3))

def prune_filters(model, prune_ratio=0.7):
    old_conv_layers = model.get_conv_layers()
    new_features = nn.Sequential()
    conv_idx = 0
    
    pruned_out_channels = []
    keep_indices_list = []
    prev_out_channels = 3
    
    # First pass: Determine pruning
    for old_layer in old_conv_layers:
        l1_norm = compute_l1_norm(old_layer.weight.data)
        num_filters = old_layer.out_channels
        num_prune = int(num_filters * prune_ratio)
        num_keep = max(num_filters - num_prune, 1)
        
        _, keep_indices = torch.topk(l1_norm, num_keep, largest=True, sorted=True)
        keep_indices = keep_indices.sort()[0]
        
        pruned_out_channels.append(num_keep)
        keep_indices_list.append(keep_indices)
    
    # Second pass: Build new features
    for i, old_layer in enumerate(old_conv_layers):
        new_layer = nn.Conv2d(
            in_channels=prev_out_channels,
            out_channels=pruned_out_channels[i],
            kernel_size=old_layer.kernel_size,
            padding=old_layer.padding,
            stride=old_layer.stride
        ).to(old_layer.weight.device)
        
        new_layer.weight.data = old_layer.weight.data[keep_indices_list[i]].clone()
        if old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data[keep_indices_list[i]].clone()
        if i > 0:
            new_layer.weight.data = new_layer.weight.data[:, keep_indices_list[i-1], :, :].clone()
        
        new_features.add_module(f"conv_{conv_idx}", new_layer)
        new_features.add_module(f"relu_{conv_idx}", nn.ReLU(inplace=True))
        
        current_pos = [j for j, m in enumerate(model.features) if isinstance(m, nn.Conv2d)][i]
        if i < len(old_conv_layers) - 1:
            next_pos = [j for j, m in enumerate(model.features) if isinstance(m, nn.Conv2d)][i + 1]
            for j in range(current_pos + 1, next_pos):
                if isinstance(model.features[j], nn.MaxPool2d):
                    new_features.add_module(f"pool_{conv_idx}", nn.MaxPool2d(kernel_size=2, stride=2))
                    break
        
        prev_out_channels = pruned_out_channels[i]
        conv_idx += 1
    
    new_model = VGG16(num_classes=10, pretrained=False)
    new_model.features = new_features
    new_model.classifier[0] = nn.Linear(prev_out_channels, 512)
    return new_model
