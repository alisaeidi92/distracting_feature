import torch
from torch import nn
import torch.nn.functional as F

from .VisionTransformer import VisionTransformer


class SimpleTransformer(nn.Module):
    def __init__(self, args):
        super(SimpleTransformer, self).__init__()

        self.transformer = VisionTransformer(
            image_size = 160,
            patch_size = 20,
            num_classes = 8,
            dim = 1024,
            depth = 6,
            heads = 3,
            mlp_dim = 2048,
            dropout = 0.1,
            channels = 16
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # print('x shape:', x.shape)
        output = self.transformer(x)
        # output = self.softmax(output)
        output = F.log_softmax(output, dim=1)
        print('simple transformer output:', output)
        return output