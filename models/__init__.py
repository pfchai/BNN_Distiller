# -*- coding: utf-8 -*-

from .resnetv2 import ResNet18
from .bin_resnetv2 import resnet18_1w1a


model_dict = {
    'resnet18': ResNet18,
    'resnet18_1w1a': resnet18_1w1a,
}
