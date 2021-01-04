import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PointGolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True, event_th = 10, npoints = 100):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.event_th = event_th
        self.npoints = npoints

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        points = []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            counter = 0
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if counter == 0:
                        base_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
                    images.append(img)
                    pt = img2event(base_img, img, th = self.event_th)
                    if len(pt) < self.npoints:
                        
                        # print(np.where(events[1:-1] == pos)[0][0])
                        choice = np.random.choice(len(post_pt), self.npoints, replace=True)
                        pt = post_pt[choice, :]
                        # pt = np.concatenate([post_pt[choice, :], pt])
                    post_pt = pt[:]
                    # print(len(pt))
                    # print(pt.shape)
                    choice = np.random.choice(len(pt), self.npoints, replace=True)
                    # exit()
                    #resample
                    pt = pt[choice, :]
                    points.append(pt)
                    base_img = img[:]
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                    counter += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images':np.asarray(images), 'labels':torch.tensor(np.asarray(labels)), "points":torch.tensor(np.array(points).transpose(0, 2, 1), dtype = torch.float)}
        # sample = {'images':np.asarray(images), 'labels':torch.tensor(np.asarray(labels), dtype = torch.float), "points":torch.tensor(np.array(points).transpose(0, 2, 1), dtype = torch.float)}
        # sample = {'images':np.asarray(images), 'labels':np.asarray(labels), "points":np.array(points).transpose(0, 2, 1)}
        # if self.transform:
        #     sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}

def img2event(base_img, img, th = 10):
    # print(img.shape)

    _img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate([img, _img_gray], axis = 2)

    _base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY).reshape(base_img.shape[0], base_img.shape[1], 1)
    base_img = np.concatenate([base_img, _base_img_gray], axis = 2)
    img3 = base_img.astype(float) - img.astype(float)
    positive = np.where(img3 > th)
    positive_points = np.array([[positive[0][i], positive[1][i],
                                 positive[2][i], 1] for i in range(len(positive[0]))])

    negative = np.where(img3 < -1 * th)
    negative_points = np.array([[negative[0][i], negative[1][i],
                                 negative[2][i], -1] for i in range(len(negative[0]))])
    
    # print(positive_points.shape, negative_points.shape)
    if len(positive_points) == 0 and len(negative_points) == 0:
        return [0,0,0,0]
    elif len(positive_points) == 0:
        return negative_points
    elif len(negative_points) == 0:
        return positive_points
    else:
        return np.concatenate([positive_points, negative_points], axis = 0)



if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))




    





       

