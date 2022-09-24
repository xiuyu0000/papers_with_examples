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
"""Prepare Helen Dataset"""

import argparse
import csv

import cv2

import numpy as np
import scipy.io as scio
from mindspore.mindrecord import FileWriter


def to_mindrocord(img_size, output_path, clip, dataset_side_data_enhance=False):
    """
    Write Helen Dataset to Mindrecord File

    Args:
        clip: Clip Picture or Not
        img_size: Compress each img to [img_size, img_size, 3]
        output_path(string): Output MindRecord File Path
        dataset_side_data_enhance(bool): Rotate image with annotations or not. Default: False

    Returns:
        No Direct Return
        But Generate mindrecord File With 2 Columns ['label', 'image'] at 'output_path'

    Examples:
        >>> to_mindrocord(192, '/mnt/Helen_192', True, True)
    """
    finalpictures, annotations = read_helen(img_size, dataset_side_data_enhance, clip)

    writer = FileWriter(file_name=output_path, shard_num=1)
    cv_schema = {"image": {"type": "float32", "shape": [img_size, img_size, 3]},
                 "label": {"type": "float32", "shape": [1, 388]}}
    writer.add_schema(cv_schema, "Face Alignment Dataset")

    data = []
    limit = 8000 if dataset_side_data_enhance == 'True' else 2000
    for i in range(limit):
        sample = {}
        sample['label'] = annotations[i]
        sample['image'] = finalpictures[i]

        data.append(sample)
        if i % 10 == 0:
            writer.write_raw_data(data)
            data = []
    if data:
        writer.write_raw_data(data)
    writer.commit()


def to_file(img_size, output_path, clip, dataset_side_data_enhance=False):
    """
    Output Clipped Image File Using Helen Dataset

    Create This Function To Directly Output Clipped Image and annotations Into Files to see If It clipped correctly.
    This function is not directly called in this project.
    Manually call this function when you need to output clipped imgs.

    Args:
        clip: Clip Picture or Not.
        img_size(int): image clipped & resized to reach this size
        output_path(string): output folder
        dataset_side_data_enhance(bool): Rotate image with annotations or not. Default: False

    Returns:
        No Direct Return
        But Generate Clipped & Resized Images and annotations

    Examples:
        >>> to_file(192, '/mnt/img', True, True)
    """
    finalpictures, annotations = read_helen(img_size, dataset_side_data_enhance, clip)
    pic_dir = output_path + "/clipped_pics/"
    anno_dir = output_path + "/clipped_annotation/"
    limit = 8000 if dataset_side_data_enhance else 2000
    for i in range(limit):
        cv2.imwrite(pic_dir + extend_file_name(i, 5) + ".jpeg", finalpictures[i])
        f = open(anno_dir + extend_file_name(i, 5) + ".txt", "a")
        for j in range(annotations[i].shape[0]):
            f.write(str(annotations[i][j][0]) + "," + str(annotations[i][j][1]) + "\n")
        f.close()


def extend_file_name(index, length):
    """
    Prepare File Name, Extend To The Same Length

    Args:
        index(int): Number
        length(int): Fill '0' before Number to Ensure File Name the Same Length

    Returns:
        String: 0-Prefixed Number String

    Example:
        >>> extend_file_name(24, 5)

    """
    index_len = len(str(index))
    return str(0) * (length - index_len) + str(index)


def read_helen(img_size, dataset_side_data_enhance=False, clip=False):
    """
    Read Helen Data In Files and Generate Dataset

    Args:
        clip: Clip Picture or Not. Default: False.
        img_size: Compress each img to [img_size, img_size, 3]
        dataset_side_data_enhance: Rotate or not. Default: False

    Returns:
        finalpictures: Array, Contain multiple Pictures in [-1, img_size, img_size, 3]
        annotations: Array, Contain multiple annotations in [-1, img_size, img_size, 3]

    """
    filename = []
    with open("Helen/trainname.txt") as file:
        for item in file:
            filename.append(item.replace("\n", ""))
    file.close()
    root_dir = "Helen/"
    bounding_box = scio.loadmat(root_dir + "bounding_boxes_helen_trainset.mat").get("bounding_boxes")[0]

    groundtruthboxes = []
    detectorboxes = []
    finalpictures = []
    annotations = []
    for i in range(0, 2000):
        assert str(filename[i] + ".jpg") == bounding_box[i][0][0][0][0]

        img_path = root_dir + "train/" + filename[i] + ".jpg"
        img = cv2.imread(img_path, flags=1)
        annotation_path = root_dir + "annotation/" + str(i + 1) + ".txt"
        annotation = read_csv(annotation_path)
        ground_truth_box = bounding_box[i][0][0][2][0].astype(np.int32)
        groundtruthboxes.append(ground_truth_box)
        detecter_box = bounding_box[i][0][0][1][0].astype(np.int32)
        detectorboxes.append(detecter_box)
        if clip:
            final_pic = picture_clip(img, ground_truth_box)
            final_pic, new_annotation = picture_resize(final_pic, annotation, ground_truth_box[0],
                                                       ground_truth_box[1], img_size)
        else:
            final_pic = img
            final_pic, new_annotation = picture_resize(final_pic, annotation, 0, 0,
                                                       img_size)
        final_pic = final_pic.astype(np.float32)
        new_annotation = new_annotation.astype(np.float32)
        if dataset_side_data_enhance == 'True':
            pic_1 = cv2.rotate(final_pic, cv2.ROTATE_90_CLOCKWISE)
            anno_1 = new_annotation.copy()
            anno_1[:, 0] = img_size - new_annotation[:, 1]
            anno_1[:, 1] = new_annotation[:, 0].copy()
            pic_2 = cv2.rotate(final_pic, cv2.ROTATE_180)
            anno_2 = new_annotation.copy()
            anno_2[:, 0] = img_size - new_annotation[:, 0]
            anno_2[:, 1] = img_size - new_annotation[:, 1]
            pic_3 = cv2.rotate(final_pic, cv2.ROTATE_90_COUNTERCLOCKWISE)
            anno_3 = new_annotation.copy()
            anno_3[:, 0] = new_annotation[:, 1].copy()
            anno_3[:, 1] = img_size - new_annotation[:, 0]
            finalpictures.append(pic_1)
            annotations.append(anno_1.astype(np.float32))
            finalpictures.append(pic_2)
            annotations.append(anno_2.astype(np.float32))
            finalpictures.append(pic_3)
            annotations.append(anno_3.astype(np.float32))
        finalpictures.append(final_pic)
        annotations.append(new_annotation.astype(np.float32))
    return finalpictures, annotations


def read_csv(path):
    """
    Read csv File
    Args :
        path(str): Helen Annotation TXT File Path

    Returns :
        result(numpy.ndarray): Annotation Data in np.ndarray. For Helen Dataset, output shape is (194, 2).
    """
    data = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    result = np.array(data[1:], dtype=float)
    return result


def picture_clip(pic, box):
    """
    Clip Image Using Bounding Box

    Input :
        pic(ndarray) : Picture at any size
        box(ndarray) : Box in [xMin,yMin,xMax,yMax]

    Output : Clipped Picture
    Example :
        >>> picture_clip(pic, [1, 5, 65, 97])
    """
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    img_crop = pic[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    return img_crop


def picture_resize(picture, annotation, x0, y0, target_size):
    """
    Resize Picture And Adjusy Annotation According to Start Point and Target Size
    Pictures should be resized, annotations need sub and 'resize'

    Input :
        Picture : CV2 Picture [ W, H, C ]
        annotation : Marked Points , Absolute Position , [(x1,y1),(x2,y2)...]
        x0 : Bounding Box's Left Upper Corner's Position on X axis
        y0 : Bounding Box's Left Upper Corner's Position on Y axis
        target_size : Will Resize Image To (target_size, target_size)

    Output :
        Picture : Resized Picture
        annotation ： annotations , But Relative Position , Relate to Resized Picture

    Examples:
        >>>picture_resize(img, annotation, 10, 20, 192)
    """
    y_ratio, x_ratio = target_size / picture.shape[0], target_size / picture.shape[1]
    img_resized = cv2.resize(picture, (target_size, target_size))
    img_resized = img_resized / 255
    annotation[:, 0] = annotation[:, 0] - x0
    annotation[:, 1] = annotation[:, 1] - y0
    annotation[:, 0] = annotation[:, 0] * x_ratio
    annotation[:, 1] = annotation[:, 1] * y_ratio

    return img_resized, annotation


def parse_args():
    """Parse configuration arguments for infer."""
    parser = argparse.ArgumentParser(description='Helen Dataset Generation')
    parser.add_argument('--img_size', type=int, default=224, help='Pretrained checkpoint path')
    parser.add_argument('--dataset_target_path', type=str, default=None, help='Helen Dataset Root Path')
    parser.add_argument('--dataset_side_data_enhance', type=bool, default=False,
                        help='Run Data Enhance On Dataset Processing')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args_opt = parse_args()
    to_mindrocord(args_opt.img_size, args_opt.dataset_target_path, False,
                  dataset_side_data_enhance=args_opt.dataset_side_data_enhance)
