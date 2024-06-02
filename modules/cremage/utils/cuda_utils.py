# CUDA utils
import torch

def gpu_total_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return torch.cuda.get_device_properties(device).total_memory
    else:
        return -1

def gpu_allocated_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return torch.cuda.memory_allocated(device)
    else:
        return -1

def gpu_reserved_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return torch.cuda.memory_reserved(device)
    else:
        return -1

def gpu_free_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return torch.cuda.get_device_properties(device).total_memory - \
            torch.cuda.memory_allocated(device)
    else:
        return -1

def gpu_memory_info():
    if torch.cuda.is_available():
        total = gpu_total_memory() / 1024**2  # TO MB
        free = gpu_free_memory() / 1024**2
        return f"{free:.0f} MB free/{total:.0f} MB total"
    else:
        return "Info not available"