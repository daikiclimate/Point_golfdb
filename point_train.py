import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from point_dataloader import PointGolfDB
from point_model import PointEventDetector
from util import *

if __name__ == "__main__":

    # training configuration
    split = 1
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    n_cpu = 6
    seq_length = 128
    # seq_length = 64
    bs = 2  # batch size
    k = 10  # frozen layers

    model = PointEventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        # lstm_hidden=1024,
        lstm_hidden=128,
        # lstm_hidden=256,
        bidirectional=True,
        dropout=False,
        feature_dim = 5
    )
    freeze_layers(k, model)
    model.train()
    model.cuda()

    dataset = PointGolfDB(
        data_file="data/train_split_{}.pkl".format(split),
        vid_dir="data/videos_160/",
        seq_length=seq_length,
        train=True,
        # npoints=2048,
        npoints=1024,
        # npoints=256,
        event_th=10,
        # event_th = 50
        # base_list=[-1,-2,-3,-4,-5],
        base_list=[-1,-5, -10, -20, -30],
    )

    data_loader = DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=n_cpu, drop_last=True
    )

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor(
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 35]
    ).cuda()
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
            points, labels = sample["points"].cuda(), sample["labels"].cuda()
            logits = model(points)
            labels = labels.view(bs * seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), points.size(0))
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
                    "models/point_swingnet_{}.pth.tar".format(i),
                )
            if i == iterations:
                break
