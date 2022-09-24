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
# ==============================================================================
"""
Exponential moving average.
"""

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

# Generate overloaded functions
_ema_op = C.MultitypeFuncGraph("grad_ema_op")
assign_add = P.AssignAdd()


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    """
    Gradient and used to accumulate gradients.

    Args:
        ema_weight (int): Exponential moving average weight.
        weight (int): Weight value used for momentum update.

    Returns:
        Tensor, the same data type and shape as original variable.
    """

    # Momentum updates applied
    return assign_add(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMACell(nn.Cell):
    """
    A copy of the exponential moving average weight model is created for alignment and updating.

    Args:
        weights (dict): Model weight.
        ema_decay (float): Decay rate used to control the speed of model update.
    """

    def __init__(self, weights, ema_decay=0.9999):
        super(EMACell, self).__init__()
        self.ema_weights = weights.clone(prefix="_ema_weights")
        self.ema_decay = Tensor(ema_decay, mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, weights):
        """
        Operate on each element of the sequence or nested sequence.

        Args:
            weights (dict): Model weight.

        Returns:
            Nested sequence, result of set operation.
        """

        # Do set operations on the input sequence
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay), self.ema_weights, weights)
        return success
