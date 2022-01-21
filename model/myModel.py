# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
一个简单的模型示例。
"""
from typing import Optional

import numpy as np
from sklearn.datasets import make_blobs

"""
MLM demo训练

"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
# from tkitAutoMask import autoMask
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.functional import precision_recall, accuracy, f1


# from transformers import BertTokenizer


class myModel(pl.LightningModule):
    """
    EncDec
    使用transformer实现
    """

    def __init__(self, learning_rate=5e-5,
                 T_max=5,
                 c=0.01,
                 optimizer_name="SGD",
                 dropout=0.2,
                 batch_size=2,
                 trainfile="./out/train.pkt",
                 valfile="./out/val.pkt",
                 testfile="./out/test.pkt",
                 T_mult=2,
                 T_0=500,
                 **kwargs):
        super().__init__()
        # save save_hyperparameters
        self.save_hyperparameters()

        self.model = nn.Linear(2, 1)
        # self.sm=nn.Sigmoid()

    def forward(self, X, Y, **kwargs):
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        # N = len(Y)
        # print(X, Y)

        # perm = torch.randperm(N)

        # for i in range(0, N, self.hparams.batch_size):
        #     x = X[perm[i : i + self.hparams.batch_size]].to(self.device)
        #     y = Y[perm[i : i + self.hparams.batch_size]].to(self.device)
        #     print(x)
        #     print(y)
        x = X
        y = Y
        x = self.model(x)
        # x=self.sm(x)
        weight = self.model.weight.squeeze()

        loss = self.loss_fc(x, y)
        return x, loss

    def loss_fc(self, x, y):
        weight = self.model.weight.squeeze()
        loss = torch.mean(torch.clamp(1 - y * x, min=0))
        loss += self.hparams.c * (weight.t() @ weight) / 2.0
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        x, y = batch
        # src, _ = self.tomask(src)

        outputs, loss = self(x, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        x, y = batch

        outputs, loss = self(x, y)
        metrics = {
            "val_loss": loss
        }
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch
        input_ids, labels = self.tomask(input_ids)
        outputs = self(input_ids.long(), token_type_ids.long(), attention_mask.long(), labels.long())
        pred = outputs.logits.argmax(-1)
        print("pred", pred)
        with open("test_ner.txt", "a+") as f:
            words = self.tokenizer.convert_ids_to_tokens(input_ids.view(-1).tolist())
            for i, (w, x, y, l, m) in enumerate(
                    zip(words, input_ids.view(-1).tolist(), pred.view(-1).tolist(), labels.view(-1).tolist(),
                        attention_mask.view(-1).tolist())):
                if m == 1:
                    print(w, x, y, l, m)
                    # f.write(str(y)+"--"+str(l))
                    # f.write(",".join([w,x,y,l,m]))
                    # f.write("\n")
                    # f.write("".join(words).replace("[PAD]", " "))
                    # f.write("\n")

        active_loss = attention_mask.view(-1) == 1
        precision, recall = precision_recall(pred.view(-1)[active_loss], labels.reshape(-1).long()[active_loss],
                                             average='macro', num_classes=self.hparams.num_labels)

        pred_f1 = f1(pred.view(-1)[active_loss], labels.reshape(-1).long()[active_loss], average='macro',
                     num_classes=self.hparams.num_labels)
        acc = accuracy(pred.view(-1)[active_loss], labels.reshape(-1).long()[active_loss])

        metrics = {
            "test_precision_macro": precision,
            "test_recall_macro": recall,
            "test_f1_macro": pred_f1,
            "test_acc": acc,
            "test_loss": outputs.loss
        }
        self.log_dict(metrics)
        return metrics

    def setup(self, stage: Optional[str] = None) -> None:
        X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
        X = (X - X.mean()) / X.std()
        Y[np.where(Y == 0)] = -1
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        data = TensorDataset(X, Y)
        self.train_data, self.val_data = random_split(data, [400, 100])
        pass

    def train_dataloader(self):
        # train = torch.load(self.hparams.trainfile)

        return DataLoader(self.train_data, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        # val = torch.load(self.hparams.valfile)
        return DataLoader(self.val_data, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True)

    def test_dataloader(self):
        # val = torch.load(self.hparams.testfile)
        return DataLoader(self.val_data, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True)

    def configure_optimizers(self):
        """优化器 自动优化器"""
        optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)

        #         使用自适应调整模型
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=500000, factor=0.8,
        #                                                        verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.T_0,
                                                                         T_mult=self.hparams.T_mult, eta_min=0,
                                                                         last_epoch=-1, verbose=False)
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': "lr_scheduler",
            'monitor': 'train_loss',  # 监听数据变化
            'strict': True,
        }
        #         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == '__main__':
    pass
