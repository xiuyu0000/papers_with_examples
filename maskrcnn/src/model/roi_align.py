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
"""MaskRcnn ROIAlign module."""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.nn import layer as L
from mindspore.common.tensor import Tensor


class ROIAlign(nn.Cell):
    """
    Extract RoI features from mulitiple feature map.

    Args:
        out_size_h (int): RoI height.
        out_size_w (int): RoI width.
        spatial_scale (int): RoI spatial scale.
        sample_num (int): RoI sample number. Default: 0.
        roi_align_mode (int): RoI align mode. Default: 1.

    Inputs:
        - **features** (Tensor) - The input features, whose shape must be :math:'(N, C, H, W)'.
        - **rois** (Tensor) - The shape is :math:'(rois_n, 5)'. With data type of float16 or float32.

    Outputs:
        Tensor, the shape is :math: '(rois_n, C, pooled_height, pooled_width)'.

    Support Platform:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> features = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> rois = Tensor(np.array([[0, 0.2, 0.3, 0.2, 0.3]]), mindspore.float32)
        >>> roi_align = ops.ROIAlign(2, 2, 0.5, 2)
        >>> output = roi_align(features, rois)
        >>> print(output)
        [[[[1.775 2.025]
           [2.275 2.525]]]]
    """
    def __init__(self, out_size_h, out_size_w, spatial_scale, sample_num=0, roi_align_mode=1):
        super(ROIAlign, self).__init__()

        self.out_size = (out_size_h, out_size_w)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.align_op = P.ROIAlign(self.out_size[0], self.out_size[1],
                                   self.spatial_scale, self.sample_num,
                                   roi_align_mode)

    def construct(self, features, rois):
        """Construct ROI Align"""
        return self.align_op(features, rois)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += \
            '(out_size={}, spatial_scale={}, sample_num={}'.format(self.out_size, self.spatial_scale, self.sample_num)
        return format_str


class SingleRoIExtractor(nn.Cell):
    """
    Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level according to its scale.

    Args:
        config (dict): Config
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        batch_size (int): Batchsize. Default: 1.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        mask (bool): Specify ROIAlign for cls or mask branch. Default: False.

    Inputs:
        - **rois** (Tensor) - The shape is :math:'(rois_n, 5)'. With data type of float16 or float32.
        - **feat1** (Tensor) - The input features, whose shape must be :math:'(N, C, H, W)'.
        - **feat2** (Tensor) - The input features, whose shape must be :math:'(N, C, H, W)'.
        - **feat3** (Tensor) - The input features, whose shape must be :math:'(N, C, H, W)'.
        - **feat4** (Tensor) - The input features, whose shape must be :math:'(N, C, H, W)'.

    Outputs:
        Tensor, the shape is :math:'(rois_n, C, pooled_height, pooled_width)'.

    Support Platform:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> fea1 = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> fea2 = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> fea3 = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> fea4 = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> rois = Tensor(np.array([[0, 0.2, 0.3, 0.2, 0.3]]), mindspore.float32)
        >>> single_roi = ops.SingleRoIExtractor(conifg, 2, 1, 2, 2, mask)
        >>> output = single_roi(rois, fea1, fea2, fea3, fea4)
    """

    def __init__(self, config, roi_layer, out_channels, featmap_strides, batch_size=1, finest_scale=56, mask=False):
        super(SingleRoIExtractor, self).__init__()
        cfg = config
        self.train_batch_size = batch_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.out_size = roi_layer.mask_out_size if mask else roi_layer.out_size
        self.mask = mask
        self.sample_num = roi_layer.sample_num
        self.roi_layers = self.build_roi_layers(self.featmap_strides)
        self.roi_layers = L.CellList(self.roi_layers)

        self.sqrt = P.Sqrt()
        self.log = P.Log()
        self.finest_scale_ = finest_scale
        self.clamp = C.clip_by_value

        self.cast = P.Cast()
        self.equal = P.Equal()
        self.select = P.Select()

        in_mode_16 = False
        self.dtype = np.float16 if in_mode_16 else np.float32
        self.ms_dtype = mstype.float16 if in_mode_16 else mstype.float32
        self.set_train_local(cfg, training=True)

    def set_train_local(self, config, training=True):
        """Set training flag."""
        self.training_local = training

        cfg = config
        # Init tensor
        roi_sample_num = cfg.num_expected_pos_stage2 if self.mask else cfg.roi_sample_num
        self.batch_size = roi_sample_num if self.training_local else cfg.rpn_max_num
        self.batch_size = self.train_batch_size*self.batch_size \
            if self.training_local else cfg.test_batch_size*self.batch_size
        self.ones = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=self.dtype))
        finest_scale = np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) * self.finest_scale_
        self.finest_scale = Tensor(finest_scale)
        self.epslion = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=self.dtype)*self.dtype(1e-6))
        self.zeros = Tensor(np.array(np.zeros((self.batch_size, 1)), dtype=np.int32))
        self.max_levels = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=np.int32)*(self.num_levels-1))
        self.twos = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) * 2)
        self.res_ = Tensor(np.array(np.zeros((self.batch_size, self.out_channels, self.out_size, self.out_size)),
                                    dtype=self.dtype))

    def num_inputs(self):
        """input number."""
        return len(self.featmap_strides)

    def log2(self, value):
        """calculate log2."""
        return self.log(value) / self.log(self.twos)

    def build_roi_layers(self, featmap_strides):
        """build ROI layers."""
        roi_layers = []
        for s in featmap_strides:
            layer_cls = ROIAlign(self.out_size, self.out_size, spatial_scale=1 / s,
                                 sample_num=self.sample_num, roi_align_mode=0)
            roi_layers.append(layer_cls)
        return roi_layers

    def _c_map_roi_levels(self, rois):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor, Level index (0-based) of each RoI, shape (k, )
        """
        scale = self.sqrt(rois[::, 3:4:1] - rois[::, 1:2:1] + self.ones) * \
            self.sqrt(rois[::, 4:5:1] - rois[::, 2:3:1] + self.ones)

        target_lvls = self.log2(scale / self.finest_scale + self.epslion)
        target_lvls = P.Floor()(target_lvls)
        target_lvls = self.cast(target_lvls, mstype.int32)
        target_lvls = self.clamp(target_lvls, self.zeros, self.max_levels)

        return target_lvls

    def construct(self, rois, feat1, feat2, feat3, feat4):
        """Construct Single RoI Extractor"""
        feats = (feat1, feat2, feat3, feat4)
        res = self.res_
        target_lvls = self._c_map_roi_levels(rois)
        for i in range(self.num_levels):
            mask = self.equal(target_lvls, P.ScalarToArray()(i))
            mask = P.Reshape()(mask, (-1, 1, 1, 1))
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            mask = \
                self.cast(P.Tile()(self.cast(mask, mstype.int32), (1, 256, self.out_size, self.out_size)), mstype.bool_)
            res = self.select(mask, roi_feats_t, res)

        return res
