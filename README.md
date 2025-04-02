# VGG-16 Pruning with L1-Norm-Capped-L1

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
   cd L1-Capped-L1
## Install dependencies:
pip install -r requirements.txt

## Usage
### Run the main script in a Python environment (e.g., Google Colab with GPU):
python main.py

**The script loads a pretrained VGG-16, prunes it iteratively (4 iterations, starting at 70% pruning ratio), fine-tunes on CIFAR-10, and saves the pruned model to Google Drive.**

## Project Structure
-L1-Capped-L1/: Python package with core functionality.

----model.py: VGG-16 model definition.

----prune.py: Pruning logic.

----utils.py: Data loading, training, and evaluation.

----metrics.py: Parameter and FLOPs counting.

-main.py: Main script to execute pruning and training.

-requirements.txt: Dependencies.

-README.md: This file.

## Results
Baseline accuracy: ~70-80% (ImageNet weights on CIFAR-10).

Post-pruning and fine-tuning: Targets ~93% accuracy, ~92.7% parameter reduction, ~75.8% FLOPs reduction (per the paper).

## License
MIT License
