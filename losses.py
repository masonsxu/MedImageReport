import torch
import torch.nn as nn


def CE(output, target):
    '''
    Output: (B,L,C)。未经过softmax的logits
    Target: (B,L)
    '''
    output = output.reshape(-1, output.shape[-1])  # (*,C)
    target = target.reshape(-1).long()  # (*)
    return nn.CrossEntropyLoss()(output, target)  # 默认size_average=True，会把B*L所有词loss平均
