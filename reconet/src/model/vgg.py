# Copyright 2022   Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
VGG for style features encoding
"""
import mindspore.nn as nn
from mindspore import load_checkpoint

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_layer(base, padding=0, pad_mode='same', has_bias=False, batch_norm=False):
    """Make stage network of VGG.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        padding (int): Conv2d padding value. Default: 0.
        pad_mode (str): Conv2d pad mode. Default: False.
        has_bias (int): Whether conv2d has bias
        batch_norm(bool): Whether vgg has batch norm layer

    Returns:
        Vgg layers
    """
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=padding,
                               pad_mode=pad_mode,
                               has_bias=has_bias)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        padding (int): Padding value. Default: 0.
        pad_mode (str): Pad mode. Default: False.
        has_bias (int): Whether conv2d has bias
        batch_norm(bool): Whether vgg has batch norm layer

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg('16')
    """

    def __init__(self,
                 edition,
                 padding=1,
                 pad_mode='pad',
                 has_bias=False,
                 batch_norm=False):
        super(Vgg, self).__init__()
        self.layers = _make_layer(cfg[edition], padding, pad_mode, has_bias, batch_norm)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        return x


class VggEncoder(Vgg):
    """
    Encode style features with VGG network

    Args:
        edition (str): Edition of vgg model
        padding (int): Padding value
        pad_mode (str): Padding model
        has_bias (bool): Whether vgg model has bias

    Returns:
        List, list of vgg encoded features
    """

    def __init__(self, edition='16', padding=0, pad_mode='same', has_bias=True):
        super(VggEncoder, self).__init__(edition=edition,
                                         padding=padding,
                                         pad_mode=pad_mode,
                                         has_bias=has_bias)

    def encode(self, x):
        """
        Get encoded style features from specific vgg layers
        For reconet, [3, 8, 15, 22] is used
        """
        layers_of_interest = [3, 8, 15, 22]
        result = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in layers_of_interest:
                result.append(x)
        return result


def vgg16(ckpt_file, edition='16', padding=0, pad_mode='same', has_bias=True):
    """
    Build vgg16 model

    Args:
        ckpt_file (str): The path of checkpoint files
        edition (str): Edition of vgg model
        padding (int): Padding value
        pad_mode (str): Padding model
        has_bias (bool): Whether vgg model has bias

    Outputs:
        Vgg16 Model
    """
    model = VggEncoder(edition, padding, pad_mode, has_bias)
    load_checkpoint(ckpt_file, net=model)
    return model
