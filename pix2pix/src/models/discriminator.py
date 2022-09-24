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
"""Define discriminator model: Patch GAN."""

import mindspore.nn as nn
from mindspore.ops import Concat

from .conv_relu_block import ConvNormRelu


class Discriminator(nn.Cell):
    """
    Discriminator of Model.

    Args:
        config (class): Option class.
        in_planes (int): Input channel. Default: 3.
        ndf (int): the number of filters in the last conv layer. Default: 64.
        n_layers (int): The number of ConvNormRelu blocks. Default: 3.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance". Default: "batch".

    Outputs:
        Tensor, output tensor of discriminator of model.
    """

    def __init__(self, config, in_planes=3, ndf=64, n_layers=3, alpha=0.2, norm_mode='batch'):
        super(Discriminator, self).__init__()
        kernel_size = 4
        layers = [
            nn.Conv2d(in_planes, ndf, kernel_size, 2, pad_mode='pad', padding=1),
            nn.LeakyReLU(config.alpha)
        ]
        nf_mult = ndf
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8) * ndf
            layers.append(ConvNormRelu(nf_mult_prev, nf_mult, config, kernel_size, 2, alpha, norm_mode, padding=1))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) * ndf
        layers.append(ConvNormRelu(nf_mult_prev, nf_mult, config, kernel_size, 1, alpha, norm_mode, padding=1))
        layers.append(nn.Conv2d(nf_mult, 1, kernel_size, 1, pad_mode='pad', padding=1))

        self.features = nn.SequentialCell(layers)
        self.concat = Concat(axis=1)

    def construct(self, x, y):
        x_y = self.concat((x, y))
        output = self.features(x_y)
        return output
