# Pytorch学习笔记
## Pytorch安装
pytorch官网地址: https://pytorch.org/get-started/locally/
mac安装命令: `pip3 install torch torchvision torchaudio`

## 测试安装是否成功
```python
import torch
import torchvision
import torchaudio
print("python version: ", torch.__version__)
print("if MacOs GPU is available: ", torch.device('mps'))
```