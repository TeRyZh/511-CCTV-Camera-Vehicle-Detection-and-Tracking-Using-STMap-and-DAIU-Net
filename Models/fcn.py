from torch import nn

from torchvision.models.segmentation.fcn import FCNHead
from torchvision import models


def FCN(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
    model.classifier = FCNHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model