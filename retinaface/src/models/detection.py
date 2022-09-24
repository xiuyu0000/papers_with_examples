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

from __future__ import print_function
import os
import datetime
import numpy as np

from src.utils import decode_bbox, decode_landm


class DetectionEngine:
    '''
    Method to decode the output of network.
    Including nms, write the result into file,compute iou value.
    '''
    def __init__(self):
        self.results = {}
        self.nms_thresh = 0.4
        self.conf_thresh = 0.2
        self.iou_thresh = 0.6
        self.var = [0.1, 0.2]
        self.save_prefix = './widerface_result'
        self.gt_dir = './dataset/ground_truth/'

    def _iou(self, a, b):
        A = a.shape[0]
        B = b.shape[0]
        max_xy = np.minimum(
            np.broadcast_to(np.expand_dims(a[:, 2:4], 1), [A, B, 2]),
            np.broadcast_to(np.expand_dims(b[:, 2:4], 0), [A, B, 2]))
        min_xy = np.maximum(
            np.broadcast_to(np.expand_dims(a[:, 0:2], 1), [A, B, 2]),
            np.broadcast_to(np.expand_dims(b[:, 0:2], 0), [A, B, 2]))
        inter = np.maximum((max_xy - min_xy + 1), np.zeros_like(max_xy - min_xy))
        inter = inter[:, :, 0] * inter[:, :, 1]

        area_a = np.broadcast_to(
            np.expand_dims(
                (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1),
            np.shape(inter))
        area_b = np.broadcast_to(
            np.expand_dims(
                (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), 0),
            np.shape(inter))
        union = area_a + area_b - inter
        return inter / union

    def _nms(self, boxes, threshold=0.5):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indices = np.where(ovr <= threshold)[0]
            order = order[indices + 1]

        return reserved_boxes

    def write_result(self):
        # save result to file.
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            if not os.path.isdir(self.save_prefix):
                os.makedirs(self.save_prefix)

            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.results, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, boxes, ldm, confs, resize, scale, ldm_scale, image_path, priors):
        if boxes.shape[0] == 0:
            # add to result
            event_name, img_name = image_path.split('/')
            self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                       'bboxes': [], 'landmarks': []}
            return

        boxes = decode_bbox(np.squeeze(boxes.asnumpy(), 0), priors, self.var)
        boxes = boxes * scale / resize

        ldm = decode_landm(np.squeeze(ldm.asnumpy(), 0), priors, self.var)
        ldm = ldm * ldm_scale / resize

        scores = np.squeeze(confs.asnumpy(), 0)[:, 1]
        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        ldm = ldm[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        ldm = ldm[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        ldm = ldm[keep, :]

        dets[:, 2:4] = (dets[:, 2:4].astype(np.int) - dets[:, 0:2].astype(np.int)).astype(np.float)  # int
        dets[:, 0:4] = dets[:, 0:4].astype(np.int).astype(np.float)  # int
        ldm[:, 0:10] = ldm[:, 0:10].astype(np.int).astype(np.float)  # int

        # add to result
        image_info = image_path.split('/')
        if image_info[-2] not in self.results.keys():
            self.results[image_info[-2]] = {}
        self.results[image_info[-2]][image_info[-1][:-4]] = {'img_path': image_path,
                                                             'bboxes': dets[:, :5].astype(np.float).tolist(),
                                                             'landmarks': ldm[:, :10].astype(np.float).tolist()}

    def _get_gt_boxes(self):
        from scipy.io import loadmat
        gt = loadmat(os.path.join(self.gt_dir, 'wider_face_val.mat'))
        hard = loadmat(os.path.join(self.gt_dir, 'wider_hard_val.mat'))
        medium = loadmat(os.path.join(self.gt_dir, 'wider_medium_val.mat'))
        easy = loadmat(os.path.join(self.gt_dir, 'wider_easy_val.mat'))

        faceboxes = gt['face_bbx_list']
        events = gt['event_list']
        files = gt['file_list']

        hard_gt_list = hard['gt_list']
        medium_gt_list = medium['gt_list']
        easy_gt_list = easy['gt_list']

        return faceboxes, events, files, hard_gt_list, medium_gt_list, easy_gt_list

    def _norm_pre_score(self):
        max_score = 0
        min_score = 1

        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float)
                if bbox.shape[0] <= 0:
                    continue
                max_score = max(max_score, np.max(bbox[:, -1]))
                min_score = min(min_score, np.min(bbox[:, -1]))

        length = max_score - min_score
        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float)
                if bbox.shape[0] <= 0:
                    continue
                bbox[:, -1] -= min_score
                bbox[:, -1] /= length
                self.results[event][name]['bboxes'] = bbox.tolist()

    def _image_eval(self, predict, gt, keep, iou_thresh, section_num):

        _predict = predict.copy()
        _gt = gt.copy()

        image_p_right = np.zeros(_predict.shape[0])
        image_gt_right = np.zeros(_gt.shape[0])
        proposal = np.ones(_predict.shape[0])

        # x1y1wh -> x1y1x2y2
        _predict[:, 2:4] = _predict[:, 0:2] + _predict[:, 2:4]
        _gt[:, 2:4] = _gt[:, 0:2] + _gt[:, 2:4]

        ious = self._iou(_predict[:, 0:4], _gt[:, 0:4])
        for i in range(_predict.shape[0]):
            gt_ious = ious[i, :]
            max_iou, max_index = gt_ious.max(), gt_ious.argmax()
            if max_iou >= iou_thresh:
                if keep[max_index] == 0:
                    image_gt_right[max_index] = -1
                    proposal[i] = -1
                elif image_gt_right[max_index] == 0:
                    image_gt_right[max_index] = 1

            right_index = np.where(image_gt_right == 1)[0]
            image_p_right[i] = len(right_index)

        image_pr = np.zeros((section_num, 2), dtype=np.float)
        for section in range(section_num):
            _thresh = 1 - (section + 1) / section_num
            over_score_index = np.where(predict[:, 4] >= _thresh)[0]
            if over_score_index.shape[0] <= 0:
                image_pr[section, 0] = 0
                image_pr[section, 1] = 0
            else:
                index = over_score_index[-1]
                p_num = len(np.where(proposal[0:(index + 1)] == 1)[0])
                image_pr[section, 0] = p_num
                image_pr[section, 1] = image_p_right[index]

        return image_pr

    def get_eval_result(self):
        self._norm_pre_score()
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self._get_gt_boxes()
        section_num = 1000
        sets = ['easy', 'medium', 'hard']
        set_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        ap_key_dict = {0: "Easy   Val AP : ", 1: "Medium Val AP : ", 2: "Hard   Val AP : ", }
        ap_dict = {}
        for _set in range(len(sets)):
            gt_list = set_gts[_set]
            count_gt = 0
            pr_curve = np.zeros((section_num, 2), dtype=np.float)
            for i, _ in enumerate(event_list):
                event = str(event_list[i][0][0])
                image_list = file_list[i][0]
                event_predict_dict = self.results[event]
                event_gt_index_list = gt_list[i][0]
                event_gt_box_list = facebox_list[i][0]

                for j, _ in enumerate(image_list):
                    imgj = image_list[j]
                    imgj0 = imgj[0]
                    imgj00 = imgj0[0]
                    strimgj00 = str(imgj00)
                    try:
                        epds = event_predict_dict[strimgj00]
                    except KeyError:
                        continue
                    epdsb = epds['bboxes']
                    # predict = np.array(event_predict_dict[str(image_list[j][0][0])]['bboxes']).astype(np.float)
                    predict = np.array(epdsb).astype(np.float)
                    gt_boxes = event_gt_box_list[j][0].astype('float')
                    keep_index = event_gt_index_list[j][0]
                    count_gt += len(keep_index)

                    if gt_boxes.shape[0] <= 0 or predict.shape[0] <= 0:
                        continue
                    keep = np.zeros(gt_boxes.shape[0])
                    if keep_index.shape[0] > 0:
                        keep[keep_index - 1] = 1

                    image_pr = self._image_eval(predict, gt_boxes, keep,
                                                iou_thresh=self.iou_thresh,
                                                section_num=section_num)
                    pr_curve += image_pr

            precision = pr_curve[:, 1] / pr_curve[:, 0]
            recall = pr_curve[:, 1] / count_gt

            precision = np.concatenate((np.array([0.]), precision, np.array([0.])))
            recall = np.concatenate((np.array([0.]), recall, np.array([1.])))
            for i in range(precision.shape[0] - 1, 0, -1):
                precision[i - 1] = np.maximum(precision[i - 1], precision[i])
            index = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])

            print(ap_key_dict[_set] + '{:.4f}'.format(ap))

        return ap_dict
