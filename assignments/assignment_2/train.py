
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
from torch import optim
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import  OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
from dlvc.utils import LabelDecrementor
from dlvc.models.resnet50_wrapper import Resnet50Wrapper

def train(args):

    oxford_pets_path = "data/"
    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            LabelDecrementor()])#,

    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            LabelDecrementor()])

    train_data = OxfordPetsCustom(root=oxford_pets_path, 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root=oxford_pets_path, 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)



    model = Resnet50Wrapper(3,pre_trained=args.trained)



    optimizer = optim.Adam(model.parameters())
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.09)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    segMetricsTrain = SegMetrics(train_data.classes_seg)
    segMetricsVal = SegMetrics(val_data.classes_seg)

    path = f"weights/fcn_resnet50_weighted_{args.trained}.pt"

    trainer = ImgSemSegTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        segMetricsTrain,
        segMetricsVal,
        train_data,
        val_data,
        device,
        5,
        path,
        batch_size=128,
        val_frequency=1
    )

    print("Training, weighted: ", args.trained)

    trainer.train()

    print("\nTraining done.\n")
    train_metrics = trainer.metrics_train
    val_metrics = trainer.metrics_val

    # datafame is a list of tuples (loss, mIoU)
    train_loss = [x[0] for x in train_metrics]
    train_mIoU = [x[1] for x in train_metrics]
    val_loss = [x[0] for x in val_metrics]
    val_mIoU = [x[1] for x in val_metrics]

    print(f"Train mIoU: {train_mIoU}")
    print(f"Val mIoU: {val_mIoU}")

    print(f"Train loss: {train_loss}")
    print(f"Val loss: {val_loss}")

    name = f"fcn_resnet50_weighted_{args.trained}_metrics.csv"
    # save as epoch, loss, mIoU
    with open(name, "w") as f:
        f.write("epoch,loss,mIoU\n")
        for i in range(len(train_loss)):
            f.write(f"{i},{train_loss[i]},{train_mIoU[i]}\n")

    f.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('--trained', type=str, default='True', help='Use pretrained weights')

    train(args)