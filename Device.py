import torch

class Device:

    def get_cpu() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
        
    def print_device() -> str:
        device = Device.get_cpu()
        print(f'Using {device} for processing\n')