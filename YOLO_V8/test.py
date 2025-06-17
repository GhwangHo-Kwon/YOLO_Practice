import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # 장착된 GPU
print(torch.__version__)  # torch 버전