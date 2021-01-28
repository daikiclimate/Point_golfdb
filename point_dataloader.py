import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from icecream import ic


class PointGolfDB(Dataset):
    def __init__(
        self,
        data_file,
        vid_dir,
        seq_length,
        train=True,
        event_th=10,
        npoints=100,
        base_list=[-5, -10, -20, -30],
        gausian = True,
        median = False,
        expand = True,
        dim = 5
    ):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        # self.transform = transform
        self.train = train
        self.event_th = event_th
        self.npoints = npoints
        self.base_list = base_list
        self.gausian = gausian
        self.median = median
        self.expand = expand

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a["events"]
        events -= events[
            0
        ]
        # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        points = []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, "{}.mp4".format(a["id"])))

        # if False:
        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1 )
            # ic(a["id"], "+",start_frame)
            # start_frame = np.random.randint(events[-1] + 1 - self.seq_length)
            # start_frame = max(0, start_frame - seq_length)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            # print(a["id"], pos)
            counter = 0
            poss = []
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.gausian:
                        img = cv2.GaussianBlur(img,(3,3),0)
                    if self.median:
                        img = cv2.medianBlur(img,5)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                    counter += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
                poss.append(pos)
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.gausian:
                    img = cv2.GaussianBlur(img,(3,3),0)
                if self.median:
                    img = cv2.medianBlur(img,5)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        points = bases_points(images, self.base_list, self.event_th)
        # resample
        for i in range(len(points)):
            # ic(len(points[i]))
            if len(points[i]) < self.npoints:
                if self.expand: 
                    _num_points = len(points[i])
                    dim = points[i][0].shape[0]
                    _padding = np.zeros([self.npoints, dim])
                    _padding[:_num_points] = points[i]
                    points[i] = _padding
                else:
                    #sampling from previous points
                    choice = np.random.choice(
                        len(points[i - 1]), self.npoints, replace=True
                    )
                    if len(points[i]) == 0:
                        points[i] = points[i - 1][choice, :]
                    else:
                        points[i] = np.concatenate([points[i - 1][choice, :], points[i]])

            choice = np.random.choice(len(points[i]), self.npoints, replace=True)
            points[i] = points[i][choice, :]
            # points[i] = points[i] - np.expand_dims(
            #     np.mean(points[i], axis=0), 0
            # )  # center
            # dist = np.max(np.sqrt(np.sum(points[i] ** 2, axis=1)), 0)
            # points[i] = points[i] / dist  # scale

        # if self.train:
        #     sf = np.random.randint(10)
        #     # start_frame = np.random.randint(events[-1] + 1 )
        #
        #     images = images[sf:sf + self.seq_length]
        #     points = points[sf:sf + self.seq_length]
        #     labels = labels[sf:sf + self.seq_length]
        #     print(images.shape)
            # print(points.shape)
        #     print(labels.shape)

        sample = {
            "images": np.asarray(images),
            "labels": torch.tensor(np.asarray(labels)),
            "points": torch.tensor(
                np.array(points).transpose(0, 2, 1), dtype=torch.float
            ),
        }
        return sample

def img2event(base_img, img, th=10, base = 5):
    _img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(
        img.shape[0], img.shape[1], 1
    )
    img = np.concatenate([img, _img_gray], axis=2)

    _base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY).reshape(
        base_img.shape[0], base_img.shape[1], 1
    )
    base_img = np.concatenate([base_img, _base_img_gray], axis=2)
    img3 = base_img.astype(float) - img.astype(float)
    positive = np.where(img3 > th)
    positive_points = np.array(
        [
            [positive[0][i], positive[1][i], positive[2][i],base, 1]
            for i in range(len(positive[0]))
        ]
    )

    negative = np.where(img3 < -1 * th)
    negative_points = np.array(
        [
            [negative[0][i], negative[1][i], negative[2][i],base, -1]
            for i in range(len(negative[0]))
        ]
    )

    if len(positive_points) == 0 and len(negative_points) == 0:
        return False
    elif len(positive_points) == 0:
        return negative_points
    elif len(negative_points) == 0:
        return positive_points
    else:
        return np.concatenate([positive_points, negative_points], axis=0)


def bases_points(imgs, base_list, th):
    image_points = []
    for i in range(len(imgs)):
        pts = []
        for base in base_list:
            event_img = img2event(imgs[i + base], imgs[i], th, base)
            if isinstance(event_img, np.ndarray):
                pts.append(event_img)
        if len(pts) != 0:
            image_points.append(np.concatenate(pts, axis=0))
        else:
            image_points.append(np.array([[0, 0, 0, 0, 0]]))
    return image_points


if __name__ == "__main__":
    dataset = PointGolfDB(
        data_file="data/train_split_1.pkl",
        vid_dir="data/videos_160/",
        seq_length=64,
        train=True,
    )
    for i in dataset:
        pass

