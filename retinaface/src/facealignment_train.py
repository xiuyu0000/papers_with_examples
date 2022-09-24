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
Training on Helen dataset

Example:
python train.py --dataset_path (mindrecord path) --device_target GPU/Ascend
python train.py --dataset_path (mindrecord path) --device_target GPU/Ascend --pre_trained (ckpt path)
"""

import argparse
import ast

import mindspore as ms
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.model import ParallelMode

from src.model.facealignment import Facealignment2d
from src.utils.facealignment_utils import Monitor, data_load, get_lr


def train(args_parsed):
    """
    Train face alignment net

    Args:
        args_parsed(dict): Contain multiple training configs.

    Raises:
        ValueError: Unsupported device_target, this happens when 'device_target' not in ['GPU', 'Ascend']

    """
    ms.set_seed(args_opt.seed)
    # Check Supported Platform
    if args_opt.device_target == "GPU":
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="GPU",
                       save_graphs=False)
        if args_opt.run_distribute:
            print("Using Run Distribute Config")
            init("nccl")
            ms.set_auto_parallel_context(device_num=get_group_size(),
                                         parallel_mode=ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
    elif args_opt.device_target == "Ascend":
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="Ascend",
                       save_graphs=False)
    else:
        raise ValueError("Unsupported device_target.")
    print("train args: ", args_parsed)
    net = Facealignment2d(output_channel=args_parsed.num_classes)
    if args_parsed.pre_trained is not None:
        param_dict = load_checkpoint(args_parsed.pre_trained)
        load_param_into_net(net, param_dict)
    loss = nn.MSELoss()
    epoch_size = args_parsed.epoch_size
    if args_parsed.run_distribute:
        dataset, count = data_load(args_parsed.dataset_path, batch_size=args_parsed.batch_size, do_train=True,
                                   count_number=True, distribute=True)
    else:
        dataset, count = data_load(args_parsed.dataset_path, batch_size=args_parsed.batch_size, do_train=True,
                                   count_number=True, distribute=False)
    print("Get " + str(count) + " Data Samples")
    step_size = dataset.get_dataset_size()
    loss_scale = ms.FixedLossScaleManager(
        args_parsed.loss_scale, drop_overflow_update=False)
    lr = ms.Tensor(get_lr(global_step=0,
                          lr_init=0,
                          lr_end=0,
                          lr_max=args_parsed.lr,
                          warmup_epochs=args_parsed.warmup_epochs,
                          total_epochs=epoch_size,
                          steps_per_epoch=step_size))
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, args_parsed.momentum,
                      args_parsed.weight_decay, args_parsed.loss_scale)
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale)
    cb = [Monitor(lr_init=lr.asnumpy())]
    if args_parsed.run_distribute and args_parsed.device_target != "CPU":
        ckpt_save_dir = args_parsed.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"
    else:
        ckpt_save_dir = args_parsed.save_checkpoint_path + "ckpt_" + "/"
    if args_parsed.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_parsed.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=args_parsed.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="FaceAlignment_2D", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


def parse_args():
    """
    Parse configuration arguments for training.

    Returns:
        args(dict): Contain multiple training configs.

    """
    parser = argparse.ArgumentParser(description='Face Alignment Train')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path, Generated MindRecord File')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--device_target', type=str, default="GPU", help='run device_target, GPU or Ascend')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Save Checkpoint or not')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=10, help='Save Checkpoint Per N Epochs')
    parser.add_argument('--keep_checkpoint_max', type=int, default=500, help='Keep How Many New Checkpoints')
    parser.add_argument('--save_checkpoint_path', type=str, default='./checkpoint', help='Save Checkpoint To Where')
    parser.add_argument('--loss_scale', type=int, default=1024, help='Loss Scale')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial Momentum Optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--warmup_epochs', type=int, default=4, help='Num of Epochs for Warming Up')
    parser.add_argument('--epoch_size', type=int, default=1000, help='Num of Epochs to Run Train')
    parser.add_argument('--num_classes', type=int, default=388, help='Num of Out Channels')
    parser.add_argument('--weight_decay', type=float, default=0.00004, help='Decay Speed Of weight')
    parser.add_argument('--seed', type=int, default=114514, help='Seed For Mindspore')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args_opt = parse_args()
    train(args_opt)
