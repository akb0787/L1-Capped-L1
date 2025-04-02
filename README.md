# VGG-16 Pruning with L1-Norm

This repository implements filter pruning for VGG-16 using L1-norm, based on the paper "Pruning filters with L1-norm and capped L1-norm for CNN compression". It uses pretrained VGG-16 weights from PyTorch’s `torchvision` and applies iterative pruning and fine-tuning on CIFAR-10.

## Features
- Pretrained VGG-16 model adapted for CIFAR-10.
- L1-norm based filter pruning.
- Iterative pruning with fine-tuning.
- Metrics for parameter count and FLOPs.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vgg16-pruning.git
   cd vgg16-pruning
