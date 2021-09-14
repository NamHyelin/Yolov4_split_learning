# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys
import glob
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    np.random.shuffle(bboxes)
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    if bboxes.shape[0] == 0:
        return bboxes, 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)

    if flip:
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    return bboxes, min_w_h


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth):
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized

        if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        else:
            cropped = np.zeros([sheight, swidth, 3])
            cropped[:, :, ] = np.mean(img, axis=(0, 1))

            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # resize
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        # flip
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        if blur:
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


class Yolo_dataset(Dataset):
    def __init__(self, label_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()

        self.cfg = cfg
        self.train = train
        # label_path = 'C:/Users/hyeli/Anaconda3/envs/hlnam/yolov4/data/dataset/label'

        truth = {}

        for i, filename in enumerate(glob.glob(os.path.join(label_path, '*.txt'))):
            with open(filename, 'r', encoding='utf-8') as f:
                truth[str(i+1).zfill(3)+'.bmp'] = []
                for line in f.readlines():
                    data = line.split(" ")
                    data.append(int(data[0]))
                    data.pop(0)
                    truth[str(i+1).zfill(3)+'.bmp'].append([float(j) for j in data])

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        img_path = self.imgs[index]
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)
        img_path = os.path.join(self.cfg.dataset_dir, img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)

        return img, bboxes

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        # img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")  #lv='level1', no='123
    >>> lv = lv.replace("level", "")     #lv='1'
    >>> no = f"{int(no):04d}"            #no='0123'
    >>> return int(lv+no)                #10123
    """
    # raise NotImplementedError("Create your own 'get_image_id' function")
    #우리꺼는 001.bmp 니까 그냥 번호 그대로 쓰면 될듯. bmp만 제거
    return int(os.path.splitext(os.path.basename(filename))[0])  #001에서 int니까 1



if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    Cfg.dataset_dir = '/mnt/e/Dataset'
    dataset = Yolo_dataset(Cfg.train_label, Cfg)
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        plt.imshow(a.astype(np.int32))
        plt.show()
