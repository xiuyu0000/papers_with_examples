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
"""MaskRcnn Init."""

from src.model.resnet50 import ResNetFea, ResidualBlockUsing
from src.model.bbox_assign_sample import BboxAssignSample
from src.model.bbox_assign_sample_stage2 import BboxAssignSampleForRcnn
from src.model.fpn_neck import FeatPyramidNeck
from src.model.proposal_generator import Proposal
from src.model.rcnn_cls import RcnnCls
from src.model.rcnn_mask import RcnnMask
from src.model.rpn import RPN
from src.model.roi_align import SingleRoIExtractor
from src.model.anchor_generator import AnchorGenerator

__all__ = [
    "ResNetFea", "BboxAssignSample", "BboxAssignSampleForRcnn",
    "FeatPyramidNeck", "Proposal", "RcnnCls", "RcnnMask",
    "RPN", "SingleRoIExtractor", "AnchorGenerator", "ResidualBlockUsing"
]
