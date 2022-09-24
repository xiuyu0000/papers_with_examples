# Copyright 2022   Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" RecoNet train script."""

import argparse

import mindspore
import mindspore.nn as nn
from mindspore import context

from model.loss import ReCoNetWithLoss
from model.reconet import ReCoNet
from model.vgg import vgg16
from dataset.dataset import load_dataset
from utils.reconet_utils import vgg_encode_image


def main(args_opt):
    """RecoNet train."""
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    # Load Monkaa dataset
    train_dataset = load_dataset(args_opt.monkaa, args_opt.flyingthings3d)
    step_size = train_dataset.get_dataset_size()
    print('dataset size is {}'.format(step_size))

    # Create model.
    reconet = ReCoNet()
    vgg_net = vgg16(args_opt.vgg_ckpt)
    style_gram_matrices = vgg_encode_image(vgg_net, args_opt.style_file)

    model = ReCoNetWithLoss(reconet,
                            vgg_net,
                            args_opt.alpha,
                            args_opt.beta,
                            args_opt.gamma,
                            args_opt.lambda_f,
                            args_opt.lambda_o)

    # adam optimizer
    optim = nn.Adam(reconet.trainable_params(), learning_rate=args_opt.learning_rate, weight_decay=0.0)

    train_net = nn.TrainOneStepCell(model, optim)

    global_step = 0
    epochs = args_opt.epochs

    # train by steps
    for epoch in range(epochs):
        for sample in train_dataset.create_dict_iterator():
            loss = train_net(sample, style_gram_matrices)

            last_iteration = global_step == step_size // 2 * epochs - 1
            if global_step % 25 == 0 or last_iteration:
                print(f"Epoch: [{epoch} / {epochs}], "
                      f"step: [{global_step} / {step_size * epochs - 1}], "
                      f"loss: {loss}")
            global_step += 1

    # save trained model
    mindspore.save_checkpoint(reconet, args_opt.output_ckpt)


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='ReCoNet train.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--vgg_ckpt', type=str, default=None, help='Path of the vgg16 check point file.')
    parser.add_argument('--style_file', required=True, default=None, help='Location of image.')
    parser.add_argument('--monkaa', type=str, default=None, help='Path of the monkaa dataset.')
    parser.add_argument('--flyingthings3d', type=str, default=None, help='Path of the flyingthings3d dataset.')
    parser.add_argument('--output_ckpt', type=str, default='./reconet.ckpt', help='Saved name for ckpt file.')
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Value of learning rate.')
    parser.add_argument("--alpha", type=float, default=1e4, help="Weight of content loss")
    parser.add_argument("--beta", type=float, default=1e5, help="Weight of style loss")
    parser.add_argument("--gamma", type=float, default=1e-5, help="Weight of total variation")
    parser.add_argument("--lambda_f", type=float, default=1e5, help="Weight of feature temporal loss")
    parser.add_argument("--lambda_o", type=float, default=2e5, help="Weight of output temporal loss")

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    main(parse_args())
