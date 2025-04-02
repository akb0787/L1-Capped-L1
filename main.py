# main.py
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronize CUDA errors

import torch
from vgg16_pruning import VGG16, prune_filters, get_cifar10_loaders, train, evaluate, count_parameters, count_flops
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_cifar10_loaders()

print("Loading pretrained VGG-16...")
model = VGG16(num_classes=10, pretrained=True)
model.to(device)
baseline_acc = evaluate(model, test_loader, device)
baseline_params = count_parameters(model)
baseline_flops = count_flops(model)
print(f"Baseline (Pretrained): Accuracy={baseline_acc:.4f}, Params={baseline_params}, FLOPs={baseline_flops}")

# Iterative pruning
prune_ratio = 0.7
current_model = model
for i in range(4):
    print(f"\nIteration {i+1}: Pruning with ratio {prune_ratio}...")
    pruned_model = prune_filters(current_model, prune_ratio=prune_ratio)
    pruned_model.to(device)
    pruned_params = count_parameters(pruned_model)
    pruned_flops = count_flops(pruned_model)
    pruned_acc_before = evaluate(pruned_model, test_loader, device)
    print(f"Before fine-tuning: Accuracy={pruned_acc_before:.4f}, Params={pruned_params}, FLOPs={pruned_flops}")
    
    train(pruned_model, train_loader, test_loader, epochs=20, lr=0.01, device=device)
    pruned_acc = evaluate(pruned_model, test_loader, device)
    print(f"After fine-tuning: Accuracy={pruned_acc:.4f}, Params={pruned_params}, FLOPs={pruned_flops}")
    
    current_model = pruned_model
    prune_ratio = min(prune_ratio + 0.05, 0.9)

param_reduction = (baseline_params - pruned_params) / baseline_params * 100
flops_reduction = (baseline_flops - pruned_flops) / baseline_flops * 100
print(f"\nFinal Results: Accuracy={pruned_acc:.4f}, Param Reduction={param_reduction:.1f}%, FLOPs Reduction={flops_reduction:.1f}%")

torch.save(pruned_model.state_dict(), "/content/drive/My Drive/vgg16_pruned_pretrained.pth")
print("Pruned model saved to Google Drive.")
