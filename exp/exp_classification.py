from torch.utils.data import random_split, DataLoader

from functools import partial
from types import SimpleNamespace

from data_provider.uea import collate_fn
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        all_data, _ = self._get_data(flag="TRAIN")

        self.args.max_seq_len = all_data.max_seq_len

        train_, vali_ = self._split_data(all_data, self.args.max_seq_len)
        self.train_loader, self.train_data = train_
        self.vali_loader, self.vali_data = vali_

        self.args.pred_len = 0
        self.args.enc_in = all_data.feature_df.shape[1]
        self.args.num_class = len(all_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _split_data(self, full_dataset, max_seq_len):
        train_ratio = 0.8
        val_ratio = 0.2
        test_ratio = 0.0

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = (
            total_size - train_size - val_size
        )  # Ensure all samples are accounted for

        lengths = [train_size, val_size, test_size]

        # For reproducibility
        generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset, _ = random_split(
            full_dataset, lengths, generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            collate_fn=partial(collate_fn, max_len=max_seq_len),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
            collate_fn=partial(collate_fn, max_len=max_seq_len),
        )

        return (
            (train_loader, train_dataset),
            (val_loader, val_dataset),
        )

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.SGD(
        #    self.model.parameters(), lr=self.args.learning_rate, momentum=0.75
        # )
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze(-1).cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        print(len(preds), len(trues))

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self.train_data, self.train_loader
        vali_data, vali_loader = self.vali_data, self.vali_loader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("Train steps: ", train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                print("\r", iter_count, end="")

                model_optim.zero_grad()

                # print("X\n", batch_x)

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)

                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            if epoch % 10 == 0:
                args = {
                    "learning_rate": self.args.learning_rate,
                    "train_epochs": self.args.train_epochs,
                    "lradj": self.args.lradj,
                }
                args = SimpleNamespace(**args)
                # adjust_learning_rate(optimizer=model_optim, epoch=epoch, args=args)
                adjust_learning_rate(model_optim, epoch, args)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # train_loss0, train_accuracy = self.vali(train_data, train_loader, criterion)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                # "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}".format(
                # "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} ({3:.3f}) Train Acc: {4:.3f} Vali Loss: {5:.3f} Vali Acc: {6:.3f}".format(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    # train_loss0,
                    # train_accuracy,
                    vali_loss,
                    val_accuracy,
                )
            )
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print("test shape:", preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("accuracy:{}".format(accuracy))
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write("accuracy:{}".format(accuracy))
        f.write("\n")
        f.write("\n")
        f.close()
        return
