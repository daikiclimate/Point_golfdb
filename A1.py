from torchvision import transforms

from dataloader import GolfDB, Normalize, ToTensor

split = 1
seq_length = 64
dataset = GolfDB(
    data_file="data/train_split_{}.pkl".format(split),
    vid_dir="data/videos_160/",
    seq_length=seq_length,
    transform=transforms.Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
    train=True,
)


print(dataset[1]["images"].shape)
