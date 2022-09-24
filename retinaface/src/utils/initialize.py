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
"""Utils used to initialize tensors."""

import math
from functools import reduce

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, HeUniform


def init_kaiming_uniform(arr_shape, a=0, nonlinearity='leaky_relu', has_bias=False):
    """
    Kaiming initialize, generate a tensor with input shape, according to He initialization, using a uniform
    distribution.

    Args:
        arr_shape (tuple): The shape of generated tensor.
        a (float): Only use to leaky_relu, decide its' negative slope.
        nonlinearity (str): Non linearity function to be used, suggest to use relu or leaky_relu.
        has_bias (bool): Whether generate bias.

    Returns:
        A tuple, its first element is generated tuple with input shape, its second element is generated bias.

    """

    def _calculate_in(arr_shape):
        """Calculate input dimension of layer."""
        dim = len(arr_shape)
        n_in = arr_shape[1]
        if dim > 2:
            counter = reduce(lambda x, y: x * y, arr_shape[2:])
            n_in *= counter
        return n_in

    weight = initializer(HeUniform(negative_slope=a, nonlinearity=nonlinearity), arr_shape, mindspore.float32)

    bias = None
    if has_bias:
        bound_bias = 1 / math.sqrt(_calculate_in(arr_shape))
        bias = np.random.uniform(-bound_bias, bound_bias, arr_shape[0:1]).astype(np.float32)
        bias = Tensor(bias)

    return weight, bias
