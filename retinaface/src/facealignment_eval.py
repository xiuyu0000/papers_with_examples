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
"""Evaluate performance on helen dataset"""

import argparse

import cv2
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, Tensor
from mindspore import dataset as ds

from src.model.facealignment import Facealignment2d


def dataload(mindrecord_path):
    """
    Load mindrecord from File

    Args:
        mindrecord_path(string): Mindrecord path

    Returns:
        dataset(dataset), dataset read from mindrecord

    Examples:
        >>> dataload('/mnt/Generated.mindrecord')
    """
    dataset = ds.MindDataset(mindrecord_path, columns_list=["image", "label"])
    count = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples in Total, Load Successful".format(count))
    return dataset


def eval_data_preprocess(dataset):
    """
    Data preprocess function for evaluate

    Args:
        dataset(mindrecord dataset): Loaded dataset

    Returns:
        data_set(mindrecord dataset), preprocessed dataset
    """
    normalize_op = ds.vision.c_transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = ds.vision.c_transforms.HWC2CHW()
    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.float32)
    trans = [normalize_op, change_swap_op, type_cast_op]
    data_set = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.batch(batch_size=1, drop_remainder=True)
    return data_set


def parse_args():
    """
    Parse configuration arguments for evaluate.

    Returns:
        Parsed multiple arguments read from CLI.

    """
    parser = argparse.ArgumentParser(description='Face Alignment Train')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path, Generated MindRecord File')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--device_target', type=str, default="GPU", help='run device_target, GPU or Ascend')
    parser.add_argument('--num_classes', type=int, default=388, help='Number of Channels')
    args = parser.parse_args()
    return args


def eval_func(args):
    """
    Evaluate face alignment net performance on Helen dataset.

    Args:
        args(dict): Multiple arguments for eval.

    Raises:
        ValueError: Unsupported device_target, this happens when 'device_target' not in ['GPU', 'Ascend']
    """
    if args.device_target == "GPU":
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="GPU",
                       save_graphs=False)
    elif args.device_target == "Ascend":
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="Ascend",
                       save_graphs=False)
    else:
        raise ValueError("Unsupported device_target.")

    channel = args.num_classes
    net = Facealignment2d(output_channel=channel)
    dataset_raw = dataload(args.dataset_path)
    param_dict = load_checkpoint(args.pre_trained)
    load_param_into_net(net, param_dict)
    model = ms.Model(net)
    i = 0
    mnes = []
    errs = []
    for item in dataset_raw.create_dict_iterator(output_numpy=True):
        img = []
        img.append(item['image'].copy())
        dataset_one = ds.GeneratorDataset(source=img, column_names=["image"])
        dataset_ready = eval_data_preprocess(dataset_one)
        output_one = []
        for item_one in dataset_ready.create_dict_iterator(output_numpy=True):
            output_one = model.predict(Tensor(item_one['image']))
        target_output = item['label'].copy().reshape((channel, 1))
        output_np = output_one.asnumpy().reshape((channel, 1))
        ion = np.abs(target_output[250] - target_output[290])
        err = np.abs(target_output - output_np)
        errs.append(np.true_divide(err, ion))
        tmp = np.sum(err)
        mne = np.true_divide(tmp, ion * channel)
        mnes.append(mne)
        print("Cur Img Index : " + str(i))
        print("ION : " + str(ion))
        print("MNE : " + str(mne))
        print("ERR : " + str(tmp))
        img[0] = img[0] * 256
        for j in range(int(channel/2)):
            cv2.circle(img[0], (int(output_np[j * 2]), int(output_np[j * 2 + 1])), 2, (0, 0, 255), 1)
        cv2.imwrite('./predict/' + str(i) + '.jpg', img[0])
        i += 1
    total_count = i * channel
    positive_1 = 0
    positive_2 = 0
    print(len(errs))
    for k in range(i):
        for l in range(channel):
            if errs[k][l] < 0.1:
                positive_1 += 1
            if errs[k][l] < 0.2:
                positive_2 += 1
    meannormerror = np.array(mnes).sum() / i
    print("AUC 0.1 precision : " + str(positive_1 / total_count))
    print("AUC 0.2 precision : " + str(positive_2 / total_count))
    print("Mean Normalized Error : " + str(meannormerror))


if __name__ == '__main__':
    args_opt = parse_args()
    eval_func(args_opt)
