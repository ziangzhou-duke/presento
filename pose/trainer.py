import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import List
import torch.utils.data as data
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import trange
from pose.utils import get_weighted_loss_weights, AverageMeter, accuracy
from pose.datasets import get_babyrobot_data, BodyFaceDataset
from pose.models import BodyFaceEmotionClassifier
from dataclasses import dataclass
import numpy as np
import logging
import sys
import os


def create_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s: %(message)s', datefmt='%Y-%m-%d-%H-%M-%S')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger


@dataclass
class TrainHistory:
    acc: float
    loss: float


class Trainer:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.out_dir, exist_ok=True)

        import time
        self.logger = create_logger('train', os.path.join(args.out_dir, f'{time.time()}.log'))
        self.logger.info(" ".join(sys.argv))  # save entire command for reproduction

        self.init_datasets()
        self.current_epoch = 0
        self.history: List[TrainHistory] = []

    def get_scaler(self):
        scaler = {}
        feats = ["bodies", "faces", "hands_right", "hands_left", ]

        for x in feats:
            all_data = np.vstack(getattr(self.train_dataset, x))

            scaler[x] = MinMaxScaler()
            scaler[x].fit(all_data)

        return scaler

    def init_datasets(self):
        data = get_babyrobot_data()
        faces, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups = data

        # assert len(bodies) == len(faces) == len(hands_right) == len(hands_left) == len(Y) == len(Y_face)

        indices = list(range(len(bodies)))
        train_idx, test_idx = train_test_split(indices, test_size=0.3)

        self.train_dataset = BodyFaceDataset(data=data, indices=train_idx, phase="train", args=self.args)
        self.test_dataset = BodyFaceDataset(data=data, indices=test_idx, phase="val", args=self.args)

        self.logger.info(f"train samples: {len(self.train_dataset):d}")
        self.logger.info(f"test samples: {len(self.test_dataset):d}")

        scaler = self.get_scaler()

        self.train_dataset.set_scaler(scaler)
        self.test_dataset.set_scaler(scaler)

        self.train_dataset.to_tensors()
        self.test_dataset.to_tensors()

        self.train_dataset.prepad()
        self.test_dataset.prepad()

    def train(self):
        batch_size = self.args.batch_size
        self.dataloader_train = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=4
        )
        self.dataloader_test = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), num_workers=4
        )
        self.model = BodyFaceEmotionClassifier(self.args).cuda()

        self._fit()
        self.plot_history()

    def plot_history(self):
        plt.close('all')

        acc = []
        loss = []
        for h in self.history:
            acc.append(h.acc)
            loss.append(h.loss)

        plt.figure(figsize=(8, 6))
        plt.plot(acc, label='Overall accuracy')
        plt.plot(loss, label='Overall loss')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy or loss")
        plt.legend()
        plt.savefig('history.png')
        plt.close('all')

    def _fit(self):
        if self.args.weighted_loss:
            self.criterion_body = nn.CrossEntropyLoss(
                weight=torch.tensor(get_weighted_loss_weights(self.train_dataset.Y_body, 7), dtype=torch.float32)
            ).cuda()
        else:
            self.criterion_body = nn.CrossEntropyLoss().cuda()

        if self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay,
                momentum=self.args.momentum
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.1)

        lr = -1.0
        for self.current_epoch in range(self.args.epochs):
            train_acc, train_loss = self.train_epoch()

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            val_top_all, p, r, f = self.eval()

            self.history.append(TrainHistory(val_top_all, train_loss))

            self.logger.info(
                f"epoch={self.current_epoch}, train_loss={train_loss}, train_acc={train_acc}, "
                f"val_acc={val_top_all}, lr={lr}")

    def train_epoch(self):
        self.model.train()

        accuracy_meter_top_all = AverageMeter()
        loss_meter = AverageMeter()

        for i, batch in enumerate(self.dataloader_train):
            body, hand_right, hand_left, length, y, y_body = (
                batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch['length'].cuda(),
                batch['label'].cuda(), batch['label_body'].cuda()
            )

            self.optimizer.zero_grad()

            out_body = self.model.forward(
                (body, hand_right, hand_left, length)
            )

            loss = self.criterion_body(out_body, y_body)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            accs = accuracy(out_body, y_body, topk=(1,))
            accuracy_meter_top_all.update(accs[0].item(), length.size(0))
            loss_meter.update(loss.item(), length.size(0))

        torch.save(self.model, os.path.join(self.args.out_dir, f'{self.current_epoch}.ckpt'))

        return accuracy_meter_top_all.avg, loss_meter.avg

    def eval(self, get_confusion_matrix=True):
        accuracy_meter_top_all = AverageMeter()

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.dataloader_test):
                body, hand_right, hand_left, length, y, y_body = (
                    batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch['length'].cuda(),
                    batch['label'].cuda(), batch['label_body'].cuda()
                )

                out_body = self.model.forward(
                    (body, hand_right, hand_left, length)
                )

                accs = accuracy(out_body, y_body, topk=(1,))

                accuracy_meter_top_all.update(accs[0].item(), length.size(0))

                p, r, f, s = precision_recall_fscore_support(
                    y_body.cpu(), out_body.detach().cpu().argmax(dim=1), average="macro"
                )

                if get_confusion_matrix:
                    conf = confusion_matrix(
                        y.cpu().numpy(), torch.argmax(out_body, dim=1).cpu().numpy(),
                        labels=range(0, self.args.num_classes)
                    )
                    self.logger.info(conf)

        return (
            accuracy_meter_top_all.avg, p * 100, r * 100, f * 100
        )
