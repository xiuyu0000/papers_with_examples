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
"""Evaluation with the test dataset."""

import argparse

import mindspore.dataset as ds
from mindspore import context
from mindspore.dataset import vision
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

from src.dataset.dataload import DatasetsWFLW, Datasets300W
from src.model.pfld import pfld_1x_68, pfld_1x_98
from src.utils.utils import map_func
from src.utils.metric import validate



def main(args):
    # Set environment
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    # Load data and model
    transform = vision.py_transforms.ToTensor()
    assert args.model_type in ['98_points', '68_points']

    # Load dataset
    if args.model_type == '68_points':
        dataset_generator = Datasets300W(args.train_file_path, transform)
        net = pfld_1x_68()

    else:
        dataset_generator = DatasetsWFLW(args.train_file_path, transform)
        net = pfld_1x_98()
    
    dataset_val = ds.GeneratorDataset(list(dataset_generator),
                                     ["img", "landmark", "attributes", "angle"])
    dataset_val = dataset_val.batch(args.val_batchsize,
                                    input_columns=["attributes"],
                                    output_columns=["weight_attribute"],
                                    per_batch_map=map_func)
    # Load model
    LoadPretrainedModel(net, args.pretrain_model_path[args.model_type]).run()

    # Validation
    net.set_train(False)
    validate(dataset_val, net)


pretain_model = {'98_points': 'https://download.mindspore.cn/vision/pfld/PFLD1X_WFLW.ckpt',
                 '68_points': 'https://download.mindspore.cn/vision/pfld/PFLD1X_300W.ckpt'}


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--device_target', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--val_batchsize', default=1, type=int)
    parser.add_argument('--model_type', default='68_points', type=str)
    parser.add_argument('--train_file_path', default='./data/300W/test_data/list.txt', type=str)
    parser.add_argument('--pretrain_model_path', default=pretain_model, type=dict)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
