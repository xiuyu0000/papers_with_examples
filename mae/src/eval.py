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
"""Evaluation with the test dataset."""

import argparse

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model.vit import FineTuneVit
from process_datasets.dataset import get_dataset
from utils.logger import get_logger
from utils.eval_engine import get_eval_engine


def main(args):
    # Initialize the environment
    local_rank = 0
    device_num = 1
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))

    # Get the validation set
    eval_dataset = get_dataset(args, is_train=False)
    per_step_size = eval_dataset.get_dataset_size()
    if args.per_step_size:
        per_step_size = args.per_step_size
    args.logger.info("Create eval dataset finish, data size:{}".format(per_step_size))

    # Instantiated models
    net = FineTuneVit(batch_size=args.batch_size, patch_size=args.patch_size,
                      image_size=args.image_size, dropout=args.dropout,
                      num_classes=args.num_classes, **args.model_config)
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # Load from validation checkpoint
    if args.use_ckpt:
        params_dict = load_checkpoint(args.use_ckpt)
        load_param_into_net(net, params_dict)

    # Define the model and start training
    model = Model(net, metrics=eval_engine.metric,
                  eval_network=eval_engine.eval_network)

    eval_engine.set_model(model)
    eval_engine.compile(sink_size=per_step_size)
    eval_engine.eval()
    output = eval_engine.get_result()
    args.logger.info('accuracy={:.6f}'.format(float(output)))

def parse_args():
    """Parameters required for importing the model"""
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--encoder_layers', default=12, type=int)
    parser.add_argument('--encoder_num_heads', default=12, type=int)
    parser.add_argument('--encoder_dim', default=768, type=int)
    parser.add_argument('--mlp_ratio', default=4, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--use_parallel', default=False, type=bool)
    parser.add_argument('--device_target',
                        default='GPU',
                        choices=['CPU', 'GPU', 'Ascend'],
                        type=str)
    parser.add_argument('--mode', default='GRAPH_MODE', type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--parallel_mode', default='DATA_PARALLEL', type=str)
    parser.add_argument('--dataset_name', default='imagenet', type=str)
    parser.add_argument('--eval_engine', default='imagenet', type=str)
    parser.add_argument('--eval_path', default='/data0/imagenet2012/eval', type=str)
    parser.add_argument('--interpolation', default='BICUBIC', type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--eval_offset', default=100, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--num_classes', default=1000, type=int)
    parser.add_argument('--per_step_size', default=0, type=int)
    parser.add_argument('--use_ckpt', default='', type=str)
    parser.add_argument('--use_label_smooth', default=1, type=int)
    parser.add_argument('--label_smooth_factor', default=0.1, type=float)
    parser.add_argument('--loss_name', default='soft_ce', type=str)
    parser.add_argument('--learning_rate', default=1.0, type=float)
    parser.add_argument('--use_dynamic_loss_scale', default=False, type=bool)
    parser.add_argument('--loss_scale', default=1024, type=int)
    parser.add_argument('--use_ema', default=False, type=bool)
    parser.add_argument('--ema_decay', default=0.9999, type=float)
    parser.add_argument('--use_global_norm', default=True, type=bool)
    parser.add_argument('--clip_gn_value', default=1.0, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--save_dir', default='./output/', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
