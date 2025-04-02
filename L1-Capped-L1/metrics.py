# vgg16_pruning/metrics.py
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size=(1, 3, 32, 32)):
    flops = 0
    x = torch.randn(input_size).to(next(model.parameters()).device)
    def conv_hook(module, input, output):
        nonlocal flops
        batch_size, in_channels, h, w = input[0].size()
        out_channels, _, kh, kw = module.weight.size()
        flops += batch_size * out_channels * in_channels * kh * kw * h * w
    def linear_hook(module, input, output):
        nonlocal flops
        in_features = module.weight.size(1)
        out_features = module.weight.size(0)
        flops += input[0].numel() * out_features
    
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return flops
