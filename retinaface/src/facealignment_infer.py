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
"""Infer with face alignment network"""

import argparse
import json
import os

import cv2
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

from src.model.facealignment import Facealignment2d
from src.utils.facealignment_utils import data_preprocess, read_dir


def resolve_json(path, output_path):
    """
    Will use boxes to clip images and output clipped images when work with retinaface.

    .. warning::
        In File system, the input path should contain infer.json, directory infer/
        File infer.json contains bounding boxes and descriptions of pictures detected by retinaface.
        Directory infer/ contains raw images.
        See more documents about this function at facealignment.ipynb - 6.3 联合推理

    Args:
        path(string): Work folder which contains infer.json.
        output_path(string): Path to save clipped images

    Returns:
        No direct returns.
        Will generate clipped files to '{path}/infer/single'.
    """
    json_file = open(path + '/infer.json', 'r', encoding='utf-8')
    description = json.load(json_file)
    counter = 0
    for x in range(len(description)):

        # For each Picture
        temp_key = list(description.keys())[x]
        img = description[temp_key]
        img_path = img['img_path']
        read_img = cv2.imread(path+"/"+img_path)
        bboxes = img['bboxes']

        for i in range(len(bboxes)):
            if bboxes[i][4] > 0.95:
                # For Each Face
                img_clipped = pic_clip(read_img, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3])
                img_resized = cv2.resize(img_clipped, (192, 192))
                cv2.imwrite(output_path + str(counter) + ".jpg", img_resized)
                counter += 1


def pic_clip(img, x, y, width, height):
    """
    Clip image.

    Args:
        img(ndarray): Input Image.
        x(int) : Position of bounding box's left upper corner on X axis.
        y(int): Position of bounding box's left upper corner on Y axis.
        width(int): Image width.
        height(int): Image height.

    Returns:
        img_clipped(ndarray), clipped images
        img_clipped(ndarray): Clipped image

    Examples:
        >>> pic_clip(image, 29, 63, 372, 128)
    """
    if x < 0:
        t0 = 0
    else:
        t0 = x
    if y < 0:
        t1 = 0
    else:
        t1 = y
    if x + width < img.shape[1]:
        t2 = x + width
    else:
        t2 = img.shape[1]
    if y + height < img.shape[0]:
        t3 = y + height
    else:
        t3 = img.shape[0]
    img_clipped = img[int(t1):int(t3), int(t0):int(t2)]
    return img_clipped


def parse_args():
    """
    Parse configuration arguments for infer.

    .. warning::
        when 'mode' is 'standalone', args should include 'clipped_path' and 'predict_path'
        when 'mode' is 'retinaface', args should include 'raw_image_path', 'clipped_path' and 'predict_path'
    """
    parser = argparse.ArgumentParser(description='Face Alignment')
    parser.add_argument('--mode', type=str, default='standalone', help='Infer Work Alone / work with Retinaface')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--device_target', type=str, default="GPU", help='run device_target, GPU or Ascend')
    parser.add_argument('--num_classes', type=int, default=388, help='Num of Out Channels')
    parser.add_argument('--raw_image_path', type=str, default=None, help='Raw Img Folder Path')
    parser.add_argument('--clipped_path', type=str, default=None, help='Clipped Picture Output Path')
    parser.add_argument('--predict_path', type=str, default=None, help='Predict Result Output Path')
    args = parser.parse_args()
    return args


def infer(args):
    """
    Infer with face alignment net

    Will generate txt files which contains predicted annotations and jpg files to show result.

    Args:
        args(dict): Multiple arguments for eval.

    Raises:
        ValueError: Unsupported device_target, this happens when 'device_target' not in ['GPU', 'Ascend'].
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

    images = read_dir(args.clipped_path)
    net = Facealignment2d(output_channel=args.num_classes)
    model = ms.Model(net)
    if args.pre_trained is not None:
        param_dict = load_checkpoint(args.pre_trained)
        load_param_into_net(net, param_dict)
    for file in images:
        image = cv2.imread(file)
        image = np.array(image)
        image = cv2.resize(image, (192, 192))
        raw_image = image.copy()
        image = image / 255
        temp_imgs = []
        temp_imgs.append(image)
        dataset_one = ms.dataset.GeneratorDataset(source=temp_imgs, column_names=["image"])
        dataset = data_preprocess(dataset_one, False, batch_size=1)
        for item_one in dataset.create_dict_iterator(output_numpy=True):
            output_one = model.predict(ms.Tensor(item_one['image']))
            result = np.array(output_one).astype(int).reshape((int(args.num_classes / 2), 2))
            np.savetxt(args.predict_path+"/"+os.path.basename(file) + "_predict.txt", result, delimiter=",")
            for idx in range(194):
                raw_image = cv2.circle(raw_image, (int(result[idx, 0]), int(result[idx, 1])), 2, (0, 0, 255), 1)
            cv2.imwrite(args.predict_path+"/"+os.path.basename(file) + "_predict.jpg", raw_image)


if __name__ == '__main__':
    args_opt = parse_args()
    if args_opt.mode == 'standalone':
        infer(args_opt)
    elif args_opt.mode == 'retinaface':
        resolve_json(args_opt.raw_image_path, args_opt.clipped_path)
        infer(args_opt)
    else:
        raise "mode not implemented"
