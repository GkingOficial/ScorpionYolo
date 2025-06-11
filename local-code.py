import torch

print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU disponível:", torch.cuda.get_device_name(0))