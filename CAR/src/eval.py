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

import os
import argparse

import numpy as np
from tqdm import tqdm
from mindspore import load_checkpoint, load_param_into_net, context

from model.downsampler import DSN
from model.edsr import EDSR
from model.block import Quantization
from plug_in.adaptive_gridsampler.gridsampler import Downsampler
from utils.metric import compute_psnr_ssim, ValidateCell
from process_dataset.dataset import DIV2KHR, build_dataset, Set5Test


def main(args):
    # Set environment
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    scale = args.scale
    benchmark = args.benchmark
    kernel_size = 3 * scale + 1

    #build net
    kernel_generation_net = DSN(k_size=kernel_size, scale=scale)
    downsampler_net = Downsampler(kernel_size)
    upscale_net = EDSR(32, 256, scale=scale)
    quant = Quantization()

    #load checkpoint
    kgn_dict = load_checkpoint(os.path.join(args.checkpoint_path, "kgn.ckpt"))
    usn_dict = load_checkpoint(os.path.join(args.checkpoint_path, "usn.ckpt"))
    load_param_into_net(kernel_generation_net, kgn_dict, strict_load=True)
    load_param_into_net(upscale_net, usn_dict, strict_load=True)
    kernel_generation_net.set_train(False)
    upscale_net.set_train(False)
    downsampler_net.set_train(False)
    quant.set_train(False)
    valid_net = ValidateCell(kernel_generation_net, upscale_net, downsampler_net, quant, scale, scale)

    #read data
    if args.target_dataset == "DIV2KHR":
        val_dataloader = build_dataset(DIV2KHR(args.img_dir, "valid"), 1, 1, False)
    else:
        val_dataloader = build_dataset(Set5Test(args.img_dir, args.target_dataset), 1, 1, False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_list = list()
    ssim_list = list()
    save_dir = args.output_dir
    for i, data in enumerate(tqdm(val_dataloader.create_dict_iterator())):
        img = data['image']
        downscaled_img, reconstructed_img = valid_net(img)
        psnr, ssim = compute_psnr_ssim(img, downscaled_img, reconstructed_img, i, save_dir, scale, benchmark)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print('Mean PSNR: {0:.2f}'.format(np.mean(psnr_list)))
    print('Mean SSIM: {0:.4f}'.format(np.mean(ssim_list)))


def parse_args():
    """
    parse arguments
    """

    parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--checkpoint_path', type=str, default='', help='path to the pre-trained model')
    parser.add_argument('--img_dir', type=str, default='', help='path to the HR images to be downscaled')
    parser.add_argument('--target_dataset', default='BSDS100', type=str)
    parser.add_argument('--scale', default=4, type=int, help='downscale factor')
    parser.add_argument('--output_dir', type=str, default='./exp_res', help='path to store results')
    parser.add_argument('--benchmark', type=bool, default=True, help='report benchmark results')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
