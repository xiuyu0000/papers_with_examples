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
"""operation for GANloss"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["MeanShift"]

class MeanShift(nn.Cell):
    """
    Meanshift operation.

    Args:
        rgb_range (int): Indicate the order of input image's rgb channels. Default: 1.
        norm_mean (tuple): Mean value for normalization of the R,G,B channels. Default: (0.485, 0.456, 0.406).
        norm_std (tuple): Standard deviation for normalization of the R,G,B channels. Default: (0.229, 0.224, 0.225).
        sign (int): The sign of meanshift operation's bias. Default: -1.

    Inputs:
        - **x** (Tensor) - The image data.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **out** (Tensor) -  The image processed by meanshift operation.
          The output has the shape (batchsize, num_channels, height, width).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> mean_shift = MeanShift()
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> out = mean_shift(hr_img)
        >>> print(out.shape)
        (16, 3, 96, 96)
    """
    def __init__(self, rgb_range=1, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), sign=-1):
        super(MeanShift, self).__init__()
        std = Tensor(norm_std, mstype.float32)
        eye = ops.Eye()
        newe = eye(3, 3, mstype.float32).view(3, 3, 1, 1)
        new_std = std.view(3, 1, 1, 1)
        weight = Tensor(newe, mstype.float32) / Tensor(new_std, mstype.float32)
        bias = sign * rgb_range * Tensor(norm_mean, mstype.float32) / std
        self.meanshift = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1,
                                   has_bias=True, weight_init=weight, bias_init=bias)

    def construct(self, x):
        out = self.meanshift(x)
        return out
