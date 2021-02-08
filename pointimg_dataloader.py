import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from icecream import ic


class PointImgGolfDB(Dataset):
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
            counter = 0
            poss = []
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        # points = bases_points(images, self.base_list, self.event_th)
        diff_images = bases_event_images(images, self.base_list, self.event_th)

        sample = {
            "images": torch.tensor(np.asarray(diff_images)).float(),
            "labels": torch.tensor(np.asarray(labels)),
        }
        return sample

def img2event_image(base_img, img, th = 10, plus = 128, minus = 255):
    img3 = base_img.astype(float) - img.astype(float)

    index1 = img3 > th
    img3[index1] = plus

    index2 = img3 < -1 * th
    img3[index2] = minus

    img3[~(index1) & ~(index2)] = 0 
    return img3.astype(np.uint8)

def bases_event_images(imgs, base_list, th):
    assert imgs[0].shape == (160, 160),"Need Gray Image"
    event_images = []
    for i in range(len(imgs)):
        point_images = []
        for base in base_list:
            # event_img = img2event_image(imgs[i + base], imgs[i], th, plus = +1, minus = -1)
            event_img = imgs[i + base] - imgs[i]
            point_images.append(event_img)
        event_images.append(np.stack(point_images, axis=0))
    return event_images


if __name__ == "__main__":
    dataset = PointImgGolfDB(
        data_file="data/train_split_1.pkl",
        vid_dir="data/videos_160/",
        seq_length=64,
        train=True,
    )
    for i in dataset:
        pass

