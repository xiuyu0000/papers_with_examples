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
"""Common image processing functions"""

import os

import cv2
import numpy as np
import mindspore
from mindspore import Tensor


def preprocessing(img, size):
    """
    Image processing.

    Args:
        img (ndarray): input image.
        size (list): image size.

    Returns:
        Processed image.
    """

    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img / 127.5 - 1.0


def rgb_to_yuv(image):
    """
    Converting images from RGB space to YUV space.

    Args:
        image (ndarray): input image.

    Returns:
        Converted image.
    """

    rgb_to_yuv_kernel = Tensor([
        [0.299, -0.14714119, 0.61497538],
        [0.587, -0.28886916, -0.51496512],
        [0.114, 0.43601035, -0.10001026]
    ], dtype=mindspore.float32)

    # Convert the pixel value range from -1-1 to 0-1.
    image = (image + 1.0) / 2.0

    yuv_img = mindspore.numpy.tensordot(
        image,
        rgb_to_yuv_kernel,
        axes=([image.ndim - 3], [0]))

    return yuv_img


def denormalize_input(images):
    """
    Convert the pixel value range from 0-1 to 0-255.

    Args:
        images (ndarray / tensor): a batch of input images.

    Returns:
        Denormalized data.
    """

    images = images * 127.5 + 127.5

    return images


def compute_data_mean(data_folder):
    """
    Compute mean of R, G, B.

    Args:
        data_folder (str): path of data.

    Returns:
        A list of channel means.

    Examples:
        >>> compute_data_mean('./dataset/photo')
    """

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    for img_file in image_files:
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[..., ::-1]  # Convert to BGR for training


def normalize_input(images):
    """
    Convert the pixel value range from 0-255 to 0-1.

    Args:
        images (ndarray): a batch of input images.

    Returns:
        Normalized data.
    """

    return images / 127.5 - 1.0


def convert_image(img, img_size):
    """
    Change the channel order, transpose and resize.

    Args:
        img (ndarray): input image.
        img_size (list): image size.

    Returns:
        Converted image.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.asarray(img)
    return img


def inverse_image(img):
    """
    Convert the pixel value range from 0-1 to 0-255.

    Args:
        img (ndarray): input image.

    Returns:
        Converted image.
    """

    img = (img.squeeze() + 1.) / 2 * 255

    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    return img


def transform(fpath):
    """
    Image normalization and transpose.
    Convert the pixel value range from 0-255 to 0-1.

    Args:
        fpath (str): path of image.

    Returns:
        Transformed image.
    """

    image = cv2.imread(fpath)[:, :, ::-1]
    image = normalize_input(image)
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)

    return image


def inverse_transform_infer(image):
    """
    Image denormalization, transpose and change channel order.
    Convert the pixel value range from 0-1 to 0-255.
    Convert the channel order from RGB to BGR.

    Args:
        image (ndarray): input image.

    Returns:
        Inverse transformed image.
    """

    image = denormalize_input(image).asnumpy()
    image = cv2.cvtColor(image[0, :, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return image
