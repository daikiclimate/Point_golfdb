import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pointimg_dataloader import PointImgGolfDB
from model import EventDetector
from util import *

if __name__ == "__main__":

    # training configuration
    split = 1
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    n_cpu = 6
    seq_length = 64
    bs = 4  # batch size
    k = 10  # frozen layers

    device = 'cuda:0'
    model = EventDetector(
        # pretrain=False,
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
        device = device,
    )

    base_list = [-1]
    event_th = 30
    dataset = PointImgGolfDB(
        data_file="data/train_split_{}.pkl".format(split),
        vid_dir="data/videos_160/",
        seq_length=seq_length,
        # transform=transforms.Compose(
        #     [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        # ),
        train=True,
        base_list=base_list,
        event_th = event_th,
        median = True,
    )

    model.cnn[0][0] = nn.Conv2d(len(base_list), 32, 3, stride=2, padding = 1, bias = False)
    model = model.to(device)
    # freeze_layers(k, model)
    model.train()
    # model.cuda()

    data_loader = DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=n_cpu, drop_last=True
    )

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor(
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 35]
    ).to(device)
    # ).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    losses = AverageMeter()

    if not os.path.exists("models"):
        os.mkdir("models")

    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample["images"].to(device), sample["labels"].to(device)
            bs, seq, c, h, w = images.shape
            # images, labels = sample["images"].cuda(), sample["labels"].cuda()
            logits = model(images)
            # logits = model(images.view(bs*seq, c, h, w))
            labels = labels.view(bs * seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print(
                "Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})".format(
                    i, loss=losses
                )
            )
            i += 1
            if i % it_save == 0:
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_state_dict": model.state_dict(),
                    },
                    "models/swingnet_{}.pth.tar".format(i),
                )
            if i == iterations:
                break
