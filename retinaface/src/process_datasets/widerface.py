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
""" Create the WiderFace dataset. """

import os
import copy
from typing import Optional

import cv2
import numpy as np
from mindspore import dataset as de

from src.process_datasets.pre_process import PreProcessor
from src.utils.detection import BboxEncoder


class WiderFace:
    """
    A source dataset that reads, parses and augments the WiderFace dataset.

    The generated dataset has four columns [image, truth,conf,landm].
    The tensor of column image is a matrix of the float32 type.
    The tensor of column truth is a matrix of the float32 type.
    The tensor of column conf is a matrix of the float32 type.
    The tensor of column landm is a scalar of the int32 type.

    Args:
        path (str): The root directory of the WiderFace dataset.
        config (dict): A dictionary contains some configuration for dataset,
            config['image_size']: scaled image size adopted by the training network
            config['match_thresh']: rate threshold of prior box and annotation box
            config['variance']: pre-set value,is used to decode the prior box to prediction box
            config['clip']: Whether the width, height and coordinates of prior boxes are guaranteed to be between
             0 and 1 when generating. Default: None.
        batch_size (int): The batch size of dataset. Default: 32.
        repeat_num (int): The repeat num of dataset. Default: 1.
        python_multiprocessing (bool): Parallelize Python operations with multiple worker processes. This option could
        be beneficial if the Python operation is computational heavy. Default: False.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default: None.
        num_parallel_workers (int): The number of subprocess used to fetch the dataset in parallel. Default: 1.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.

    Examples:
        >>> config = {
        >>> 'variance': [0.1, 0.2],
        >>> 'clip': False,
        >>> 'image_size': 640,
        >>> 'match_thresh': 0.35,
        >>> }
        >>> from src.process_datasets.widerface import WiderFace
        >>> ds_train = WiderFace(path='./train/label.txt', config=config, batch_size=8, repeat_num=1, shuffle=True,
        >>>                 num_parallel_workers=1, num_shards=1, shard_id=0, python_multiprocessing=True).run()

    Citation:
        @inproceedings{yang2016wider,
        Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Title = {WIDER FACE: A Face Detection Benchmark},
        Year = {2016}}
    """

    def __init__(self,
                 path: str,
                 config: dict = None,
                 batch_size: int = 32,
                 repeat_num: int = 1,
                 python_multiprocessing: bool = False,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[int] = None,
                 shard_id: Optional[int] = None):
        self.batch_size = batch_size
        self.repeat_num = repeat_num
        self.multiprocessing = python_multiprocessing
        self.num_parallel_workers = num_parallel_workers
        self.config = config
        self.parse_widerface = ParseWiderFace(label_path=path)
        if num_shards == 1:
            de_dataset = de.GeneratorDataset(self.parse_widerface, ["image", "annotation"],
                                             shuffle=shuffle,
                                             num_parallel_workers=num_parallel_workers)
        else:
            de_dataset = de.GeneratorDataset(self.parse_widerface, ["image", "annotation"],
                                             shuffle=shuffle,
                                             num_parallel_workers=num_parallel_workers,
                                             num_shards=num_shards,
                                             shard_id=shard_id)
        self.dataset = de_dataset

    def run(self):
        """Fetch data."""
        aug = PreProcessor(self.config['image_size'])
        encode = BboxEncoder(self.config)

        def union_data(image, annot):
            cv2.setNumThreads(2)
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = cv2.imread(image.tostring().decode("utf-8"))
            labels = annot
            anns = np.zeros((0, 15))
            if labels.shape[0] <= 0:
                return anns
            for _, label in enumerate(labels):
                ann = np.zeros((1, 15))
                ann[0, 0:2] = label[0:2]
                ann[0, 2:4] = label[0:2] + label[2:4]
                ann[0, 4:14] = label[[4, 5, 7, 8, 10, 11, 13, 14, 16, 17]]
                if (ann[0, 4] < 0):
                    ann[0, 14] = -1
                else:
                    ann[0, 14] = 1
                anns = np.append(anns, ann, axis=0)
            target = np.array(anns).astype(np.float32)
            img, target = aug(img, target)
            out = encode(img, target)
            return out

        de_dataset = self.dataset.map(input_columns=["image", "annotation"],
                                      output_columns=["image", "truths", "conf", "landm"],
                                      column_order=["image", "truths", "conf", "landm"],
                                      operations=union_data,
                                      python_multiprocessing=self.multiprocessing,
                                      num_parallel_workers=self.num_parallel_workers)
        de_dataset = de_dataset.batch(self.batch_size, drop_remainder=True)
        de_dataset = de_dataset.repeat(self.repeat_num)
        return de_dataset


class ParseWiderFace:
    """
    Parse WiderFace dataset.

    Args:
        label_path (str): The root path of the WiderFace dataset.

    Raises:
        RuntimeError: If image path is not exists.
    """

    def __init__(self, label_path):
        super(ParseWiderFace, self).__init__()
        self.images_list = []
        self.labels_list = []
        f = open(label_path, 'r')
        lines = f.readlines()
        first = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if first is True:
                    first = False
                else:
                    c_labels = copy.deepcopy(labels)
                    self.labels_list.append(c_labels)
                    labels.clear()

                # remove '# '
                path = line[2:]

                # get image path
                path = label_path.replace('label.txt', 'images/') + path

                if not os.path.exists(path):
                    raise RuntimeError('image path is not exists.')

                self.images_list.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        # add the last label
        self.labels_list.append(labels)

        # del bbox which width is zero or height is zero
        for i in range(len(self.labels_list) - 1, -1, -1):
            labels = self.labels_list[i]
            for j in range(len(labels) - 1, -1, -1):
                label = labels[j]
                if label[2] <= 0 or label[3] <= 0:
                    labels.pop(j)
            if not labels:
                self.images_list.pop(i)
                self.labels_list.pop(i)
            else:
                self.labels_list[i] = labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        """
        Get image path and label from dataset by index.

        Args:
            item(int): Represents the index of image path or label.

        Returns:
            A tuple, its first element is a string represents path of the image, its second element is a list with
            length N, represents N annotation of boxes and landmarks in the image.
        """
        return self.images_list[item], self.labels_list[item]


def image_transform(img, val_origin_size, size1, size2):
    """
    Transform image into a specific size in order to fit the network.

    Args:
        img (numpy.ndrray): Numpy.ndarray of images, usually get from cv2.imread, a [H,W,C] shape array.
        val_origin_size (bool): Whether to evaluate the origin size image, if True, all images will be fill to the same
            size as the input size, size1 is the height and size2 is the width of specific size.If False, size1 will be
            a target size, size2 will be the max size, image will first resize to target size, if height or width of the
             image is too large, it will be then resize to max_size.
        size1 (int): If val_origin_size is True, it will be the target height of image, else it will be the target size
        of image.
        size2 (int): If val_origin_size is True, it will be the target width of image, else it will be the max size
        of image.

    Returns:
        A tuple, its first element is the image after resize, its second element is the multiple of resize.

    Raises:
        RuntimeError: If the height and width of input images do not meet requirements.
    """
    if val_origin_size:
        h_max, w_max = size1, size2
        resize = 1
        if not (img.shape[0] <= h_max and img.shape[1] <= w_max):
            raise RuntimeError('The height and width of input images do not meet requirements.')
        image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
    else:
        target_size, max_size = size1, size2
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        if not (img.shape[0] <= max_size and img.shape[1] <= max_size):
            raise RuntimeError('The height and width of input images do not meet requirements.')
        image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
    return img, resize
