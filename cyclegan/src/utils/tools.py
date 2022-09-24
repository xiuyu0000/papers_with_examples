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
"""Utils for cyclegan."""

import random
import numpy as np
from PIL import Image

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images,
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """
        Initialize the ImagePool class.

        Args:
            pool_size (int): the size of image buffer, if pool_size=0, no buffer will be created.
        """

        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Return an image from the pool.

        Args:
            images(numpy array / Tensor): the latest generated images from the generator.

        Returns:
            images Tensor from the buffer.
            By 50/100, the buffer will return input images.
            By 50/100, the buffer will return images previously stored in the buffer,
            and insert the current images to the buffer.
        """

        if isinstance(images, Tensor):
            images = images.asnumpy()
        if self.pool_size == 0:
            return Tensor(images)
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = np.array(return_images)
        if len(return_images.shape) != 4:
            raise ValueError("img should be 4d, but get shape {}".format(return_images.shape))
        return Tensor(return_images)


def save_image(img, img_path):
    """
    Save a numpy image to the disk.

    Args:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """

    if isinstance(img, Tensor):
        img = img.asnumpy()
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))
    img = decode_image(img)
    img_pil = Image.fromarray(img)
    img_pil.save(img_path)


def decode_image(img):
    """
    Decode a [1, C, H, W] Tensor to image numpy array.

    Args:
        img (numpy array / Tensor): image to decode.
    """

    mean = 0.5 * 255
    std = 0.5 * 255
    return (img[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))


def get_lr(args):
    """
    Learning rate generator.

    Args:
        args (class): option class.
    """

    if args.lr_policy == 'linear':
        lrs = [args.lr] * args.dataset_size * args.n_epochs
        lr_epoch = 0
        for epoch in range(args.n_epochs_decay):
            lr_epoch = args.lr * (args.n_epochs_decay - epoch) / args.n_epochs_decay
            lrs += [lr_epoch] * args.dataset_size
        lrs += [lr_epoch] * args.dataset_size * (args.max_epoch - args.n_epochs_decay - args.n_epochs)
        return Tensor(np.array(lrs).astype(np.float32))
    return args.lr


def load_ckpt(args, g_a, g_b, d_a=None, d_b=None):
    """
    Load parameter from checkpoint.

    Args:
        args (class): option class.
        g_a(str): path of generator_a ckpt.
        g_b(str): path of generator_b ckpt.
        d_a(str): path of discriminator_b ckpt. Default:None.
        d_b(str): path of discriminator_b ckpt. Default:None.
    """

    if args.g_a_ckpt is not None:
        param_ga = load_checkpoint(args.g_a_ckpt)
        load_param_into_net(g_a, param_ga)
    if args.g_b_ckpt is not None:
        param_gb = load_checkpoint(args.g_b_ckpt)
        load_param_into_net(g_b, param_gb)
    if d_a is not None and args.d_a_ckpt is not None:
        param_da = load_checkpoint(args.d_a_ckpt)
        load_param_into_net(d_a, param_da)
    if d_b is not None and args.d_b_ckpt is not None:
        param_db = load_checkpoint(args.d_b_ckpt)
        load_param_into_net(d_b, param_db)


def enable_batch_statistics(net):
    """
    Enable batch statistics in all BatchNorms.

    Args:
        net(Cell): network to enable batch statistics.
    """

    if isinstance(net, nn.BatchNorm2d):
        net.use_batch_statistics = True
    else:
        for cell in net.cells():
            enable_batch_statistics(cell)
