from ctypes import ArgumentError
import torch.cuda as gpu
from torch import device
import torch

def try_idle_gpu():
    if gpu.is_available():
        gpu_num = gpu.device_count()
        for id in range(gpu_num):
            if gpu.list_gpu_processes(id).split("\n")[1].startswith("no"):
                return "cuda:" + str(id)
        return cpu()
    else:
        return cpu()

def try_first_gpu():
    if gpu.is_available():
        gpu_num = gpu.device_count()
        if gpu_num > 0:
            return "cuda:" + str(0)
        return cpu()
    else:
        return cpu()

def try_second_gpu():
    if gpu.is_available():
        gpu_num = gpu.device_count()
        if gpu_num > 1:
            return "cuda:" + str(1)
        return cpu()
    else:
        return cpu()

def cpu():
    return "cpu"

def try_device(mode):
    if mode == "idle":
        return device(try_idle_gpu())
    elif mode == "first":
        return device(try_first_gpu())
    elif mode == "second":
        return device(try_second_gpu())
    else:
        raise ArgumentError(mode + " is not in [idle|first].")

if __name__ == "__main__":
    a = torch.tensor([1, 2, 3], dtype=torch.float64)
    print(a.device)

    b = a.to(try_gpu())
    print(a.device)
    print(b.device)

    a /= 2
    b *= 2
    print(a)
    print(b)