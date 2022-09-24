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
Utils For Face Alignment Related Function:
Learning rate generator
training monitor
oss Function
Data Preprocess
read_dir
"""

import math
import time
import os

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.train.callback import Callback
from mindspore.communication.management import get_rank, get_group_size


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Summary.

    Generate learning rate array

    Args:
        global_step(int): total steps of the training.
        lr_init(float): init learning rate.
        lr_end(float): end learning rate.
        lr_max(float): max learning rate.
        warmup_epochs(int): number of warmup epochs.
        total_epochs(int): total epoch of training.
        steps_per_epoch(int): steps of one epoch, value is dataset.get_dataset_size().
        global_step(int): Total steps of the training
        lr_init(float): Init learning rate
        lr_end(float): End learning rate
        lr_max(float): Max learning rate
        warmup_epochs(int): Number of warmup epochs
        total_epochs(int): Total epoch of training
        steps_per_epoch(int): Steps of one epoch, value is dataset.get_dataset_size()

    Returns:
        np.ndarray, learning rate array.

    Examples:
        >>> get_lr(0, 0, 0, 0.0001, 4, 1000, 8000)

    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy.ndarray): Train learning rate.

    Examples:
        >>> Monitor(100,lr_init=ms.Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self):
        """ Reset loss array and timer"""
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """ Calculate epoch time and epoch average loss"""
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self):
        """ Record step time"""
        self.step_time = time.time()

    def step_end(self, run_context):
        """ Calculate step time and step average loss"""
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], ms.Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, ms.Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


def data_load(mindrecord_path, do_train, batch_size=1, repeat_num=1, count_number=False, distribute=False,
              num_worker=4, shuffle=None):
    """
    Load dataset from mindrecord file.

    Args:
        mindrecord_path(string): File path of the mindrecord.
        do_train(bool): Call this function for train or not.
        batch_size(int): Dataset batch size. Default: 1.
        repeat_num(int): How many times does dataset duplicate. Default: 1.
        count_number(bool): Calculate number of items in dataset, used when get_dataset_size() is down. Default: False.
        distribute(bool): Run distributed or not. Default: False.
        num_worker(int): Number of dataset preparer. Default: 4.
        shuffle(bool): Shuffle or not. Default: None.

    Return:
        Dataset read from mindrecord file

    Examples:
        >>> ds=data_load('/mnt/dataset.mindrecord',True)
    """

    if distribute:

        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    print("Rank_id : " + str(rank_id))
    print("Device_num : " + str(device_num))

    if device_num == 1:
        dataset = ds.MindDataset(mindrecord_path,
                                 columns_list=["image", "label"],
                                 num_parallel_workers=num_worker,
                                 shuffle=shuffle)
    else:
        print("Running Distributed Dataset")
        dataset = ds.MindDataset(mindrecord_path,
                                 columns_list=["image", "label"],
                                 num_parallel_workers=num_worker,
                                 shuffle=shuffle,
                                 num_shards=device_num,
                                 shard_id=rank_id)

    count = 0
    if count_number:
        print("Calculating Size")
        count = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            count += 1
        print("Got {} samples in Total, Load Successful".format(count))

    buffer_size = 1000
    normalize_op = ds.vision.c_transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = ds.vision.c_transforms.HWC2CHW()
    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.float32)
    if do_train:
        trans = [normalize_op, change_swap_op, type_cast_op]
        dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=num_worker)
        dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_worker)
    else:
        trans = [normalize_op, change_swap_op, type_cast_op]
        dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=num_worker)

    # apply shuffle operations
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_num)

    return dataset, count


def data_preprocess(data_set, do_train, batch_size=1, repeat_num=1):
    """
    Define How images in mindrecord is processes.

    .. warning::
        Deprecated, this part has been merged to data_load() function above.

    Args:
        data_set(mindrecord dataset): dataset object.
        do_train(bool): is training or not.
        batch_size: batch_size. Default: 1.
        repeat_num: How many times does dataset duplicate. Default: 1.

    Return:
        preprocessed dataset, can be used to train

    Examples:
        >>> data_preprocess(ds, True, batch_size=8, repeat_num=2)
    """

    buffer_size = 1000
    normalize_op = ds.vision.c_transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = ds.vision.c_transforms.HWC2CHW()
    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.float32)
    if do_train:
        trans = [normalize_op, change_swap_op, type_cast_op]
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    else:
        trans = [normalize_op, change_swap_op, type_cast_op]
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


def read_dir(dir_path):
    """
    Read images in directory

    Args:
        dir_path(string): Target directory contain pictures.

    Returns:
        all_files(file array), contains image file paths.

    Examples:
        >>> files = read_dir('/mnt/example')

    """
    if dir_path[-1] == '/':
        raise "Do not tail with /"
    all_files = []
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        for f in file_list:
            f = dir_path + '/' + f
            if os.path.isdir(f):
                sub_files = read_dir(f)
                # Load File Inside Child Folder
                all_files = sub_files + all_files
            else:
                if os.path.splitext(f)[1] in ['.jpg', '.png', '.bmp', '.jpeg']:
                    all_files.append(f)
    else:
        raise "Error,not a dir"
    return all_files
