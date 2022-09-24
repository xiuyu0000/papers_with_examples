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
"""ICT Upsample infer."""

import os
import argparse

import mindspore
import mindspore.ops.operations as P
from mindspore import context

from src.networks import Generator
from src.dataset import load_dataset
from src.utils import Config, postprocess, imsave
from src.metrics import PSNR


def main(opts):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    config = Config(mode=opts.mode, condition_num=opts.condition_num,
                    image_size=opts.image_size, prior_size=opts.prior_size, mask=opts.mask_type)
    generator = Generator()
    generator.set_train(False)
    gen_path = os.path.join(opts.ckpt_path, 'InpaintingModel_gen.ckpt')
    if os.path.exists(gen_path):
        print('Strat loading the model parameters from %s' % (gen_path))
        checkpoint = mindspore.load_checkpoint(gen_path)
        mindspore.load_param_into_net(generator, checkpoint)
        print('Finished load the model')
    psnr_func = PSNR(255.0)
    test_dataset = load_dataset(config, image_flist=opts.input, edge_flist=opts.prior, mask_filst=opts.mask,
                                augment=False, training=False)
    if opts.visualize_all:
        test_batch_size = opts.sample_num
    else:
        test_batch_size = 1
    test_dataset = test_dataset.batch(test_batch_size)
    index = 0
    for sample in test_dataset.create_dict_iterator():
        name = sample['name'].asnumpy()[0]
        images = sample['images']
        edges = sample['edges']
        masks = sample['masks']
        index += test_batch_size
        outputs = generator(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        psnr = psnr_func(postprocess(images), postprocess(outputs_merged))
        mae = (P.ReduceSum()(P.Abs()(images - outputs_merged)) / P.ReduceSum()(images))
        print(psnr)
        print(mae)
        output = postprocess(outputs_merged)[0]
        path = os.path.join(opts.save_path, name[:-4] + "_%d" % (index % opts.condition_num) + '.png')
        imsave(output, path)


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='model checkpoints path')
    parser.add_argument('--save_path', type=str, help='the path of save result')
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--prior', type=str, help='path to the edges directory or an edge file')
    parser.add_argument('--output', type=str, help='path to the output directory')
    parser.add_argument('--mode', type=int, help='1:train, 2:test')
    parser.add_argument('--mask_type', type=int)
    parser.add_argument('--image_size', type=int, default=256, help='the size of origin image')
    parser.add_argument('--prior_size', type=int, default=32, help='the size of prior image from transformer')
    parser.add_argument('--same_face', action='store_true', help='Same face will be saved in one batch')

    parser.add_argument('--test_batch_size', type=int, default=8, help='equals to the condition number')
    parser.add_argument('--merge', action='store_true', help='merge the unmasked region')

    parser.add_argument('--condition_num', type=int, default=8, help='Use how many BERT output')
    parser.add_argument("--visualize_all", action='store_true', help='show the diverse results in one row')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
