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
Loss used to determine the closeness between the actual output and the expected output.
"""

from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore import ops as P


class CrossEntropySmooth(LossBase):
    """
    Used to specify the cross-entropy formula with label smoothing.

    Args:
        sparse (bool): Specifies whether to use sparse format for the target value.
        reduction (str): Specifies the calculation method to be applied to the output.
        The values are "mean", "sum", or "none".
        smooth_factor (float): Smooth factor.
        num_classes (int): Number of class.
        aux_factor (float): Aux factor.
    """

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000, aux_factor=0.4):
        super().__init__()
        self.aux_factor = aux_factor
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.shape = P.Shape()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logits, label):
        """
        One-hot coding is used to obtain the softmax cross entropy between the predicted value and the reality.

        Args:
            logits (Tensor): Enter predicted value.
            label (Tensor): Enter true value.

        Returns:
            Tuple, loss.
        """
        if isinstance(logits, tuple):
            logit, aux_logit = logits
        else:
            logit, aux_logit = logits, None

        if self.sparse:
            label = self.onehot(label, self.shape(logit)[1], self.on_value, self.off_value)

        loss = self.ce(logit, label)
        if aux_logit is not None:
            loss = loss + self.aux_factor * self.ce(aux_logit, label)
        return loss


class CrossEntropySmoothMixUp(LossBase):
    """
    Cross-entropy formula for smoothing with MixUp specified band labels.

    Args:
        reduction (str): Specifies the calculation method to be applied to the output.
        The values are "mean", "sum", or "none".
        smooth_factor (float): Smooth factor.
        num_classes (int): Number of class.

    Returns:
        Tensor, loss.
    """

    def __init__(self, reduction='mean', smooth_factor=0., num_classes=1000):
        super().__init__()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = 1.0 * smooth_factor / (num_classes - 2)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        """Build loss."""
        off_label = P.Select()(P.Equal()(label, 0.0),
                               P.Fill()(mstype.float32, P.Shape()(label), self.off_value),
                               P.Fill()(mstype.float32, P.Shape()(label), 0.0))

        label = self.on_value * label + off_label
        loss = self.cross_entropy(logit, label)
        return loss


class SoftTargetCrossEntropy(LossBase):
    """
    Soft cross entropy loss function for multi-classification problems.
    Instead of taking only one of the values after the unique heat coding,
    each one is weighted with a certain weight to calculate the final result.

    Args:
        reduction (str): Specifies the calculation method to be applied to the output.
        The values are "mean", "sum", or "none".

    Returns:
        Tensor, loss.
    """

    def __init__(self, reduction='mean'):
        super(SoftTargetCrossEntropy, self).__init__(reduction)
        self.sum = P.ReduceSum(False)
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.softmax = nn.LogSoftmax(axis=-1)

    def construct(self, logit, label):
        """Build loss."""
        logit = self.cast(logit, mstype.float32)
        label = self.cast(label, mstype.float32)
        x = self.softmax(logit)
        x = self.mul(-label, x)
        x = self.sum(x, -1)
        return self.get_loss(x)


class CrossEntropyIgnore(LossBase):
    """
    Cross entropy ignore.

    Args:
        num_classes (int): Number of class.
        ignore_label (int): Number of ignore label.

    Returns:
        Tensor, loss.
    """

    def __init__(self, num_classes=21, ignore_label=255):
        super().__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """Build loss."""
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


def get_loss(loss_name, args):
    """
    Get the final calculation of the corresponding different categories of loss.

    Args:
        loss_name (str): Loss name.
        args (buffer): Parameters.

    Returns:
        Tensor, loss.
    """
    loss = None
    if loss_name == 'ce_smooth':
        loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                                  num_classes=args.num_classes,
                                  aux_factor=args.aux_factor)
    elif loss_name == 'ce_smooth_mixup':
        loss = CrossEntropySmoothMixUp(smooth_factor=args.label_smooth_factor,
                                       num_classes=args.num_classes)
    elif loss_name == 'ce_ignore':
        loss = CrossEntropyIgnore(num_classes=args.num_classes,
                                  ignore_label=args.ignore_label)
    elif loss_name == 'soft_ce':
        loss = SoftTargetCrossEntropy()
    else:
        raise NotImplementedError

    return loss
