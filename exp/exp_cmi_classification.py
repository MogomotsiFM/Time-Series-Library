from exp.exp_classification import Exp_Classification
from data_provider.data_loader import CMILoader

from torch.utils.data import random_split, DataLoader

from functools import partial
from typing import Union
from types import SimpleNamespace

from data_provider.uea import collate_fn, Normalizer
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

from typing import List, Literal
from typing_extensions import override

warnings.filterwarnings("ignore")


class Exp_CMI_Classification(Exp_Classification):
    def __init__(self, args):
        super().__init__(args)

    @override
    def _build_model(self):
        normalizer = Normalizer()
        self.args.test_seq_ids = set()

        # model input depends on data
        self.train_data, self.train_loader = self._get_data(
            flag="TRAIN", normalizer=normalizer
        )
        self.vali_data, self.vali_loader = self._get_data(
            flag="VALI", max_seq_len=self.train_data.max_seq_len, normalizer=normalizer
        )

        self.args.max_seq_len = self.train_data.max_seq_len
        self.args.seq_len = self.train_data.max_seq_len

        self.args.pred_len = 0
        self.args.enc_in = self.train_data.feature_df.shape[1]
        self.args.num_class = len(self.train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    @override
    def _get_data(
        self, flag, max_seq_len=0, normalizer: Union[Normalizer, None] = None
    ):
        shuffle_flag = False if (flag == "vali" or flag == "VALI") else True
        # batch_size = 1 if (flag == "vali" or flag == "VALI") else self.args.batch_size
        batch_size = self.args.batch_size

        data_set = CMILoader(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            # limit_size=100 * self.args.batch_size,
            flag=flag,
        )

        max_seq_len = max(max_seq_len, data_set.max_seq_len)
        print(type(self), f"Max seq len-{flag}: ", max_seq_len)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=True,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            collate_fn=partial(collate_fn, max_len=max_seq_len),
        )
        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        # Window
        total_win_loss = []
        all_win_preds = []
        all_win_trues = []

        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label0, padding_mask) in enumerate(vali_loader):
                windowed_preds = []
                for end in range(self.args.max_seq_len, vali_data.max_seq_len):
                    start = end - self.args.max_seq_len

                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label0.to(self.device)

                    b = batch_x[:, start:end, :]
                    m = padding_mask[:, start:end]  # batch, max_seq_len

                    # This ensures that each sequence in a batch is used at least once
                    # A better solution is to use the subsequence that yields the highest score
                    # print("s: ", start)
                    # if True or start > 0:
                    seq_len = torch.sum(m, dim=1)  # batch, seq_len => batch
                    min_seq_len_mask = torch.greater_equal(
                        seq_len, 0.5 * self.args.max_seq_len
                    )

                    min_seq_len_mask = torch.logical_or(
                        min_seq_len_mask, torch.tensor([start]) == 0
                    )

                    if not torch.any(min_seq_len_mask):
                        break

                    outputs = self.model(b, m, None, None)

                    windowed_preds.append(
                        torch.softmax(
                            torch.reshape(outputs, (-1, 1, self.args.c_out)),
                            dim=-1,
                        )
                    )

                    outputs = outputs[min_seq_len_mask]

                    pred = outputs.detach().cpu()

                    label = label[min_seq_len_mask]
                    # else:
                    #    outputs = self.model(b, m, None, None)

                    #    pred = outputs.detach().cpu()

                    loss = criterion(pred, label.long().squeeze(-1).cpu())
                    total_loss.append(loss)

                    preds.append(outputs.detach())
                    trues.append(label)

                pred0 = Exp_CMI_Classification.select_best_predictions(
                    windowed_preds, strategy="max"
                )
                loss0 = criterion(pred0, label0.long().squeeze(-1).cpu())
                total_win_loss.append(loss0)

                all_win_preds.append(pred0)
                all_win_trues.append(label0)

        total_loss = np.average(total_loss)
        total_win_loss = np.average(total_win_loss)
        # print("Average windowed loss: ", total_win_loss)

        all_win_preds = torch.cat(all_win_preds, dim=0)
        all_win_trues = torch.cat(all_win_trues, dim=0)
        predictions = torch.argmax(all_win_preds, dim=1).cpu().numpy()
        all_win_trues = all_win_trues.flatten().cpu().numpy()
        win_accuracy = cal_accuracy(predictions, all_win_trues)
        # print("Windowed accuracy: ", win_accuracy)

        print(len(preds), len(trues))
        print(preds[0].shape, trues[0].shape)

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
        # return total_loss, accuracy
        return total_win_loss, win_accuracy

    def test(self, setting, test=0):
        pass

    @staticmethod
    def select_best_predictions(
        windowed_preds: List[torch.Tensor], strategy: Literal["max", "mode"] = "mode"
    ):
        """
        For validation purposes
        During validation we scan a window of size max_seq_len across each of the samples and pass it through
        the model. Each of these windows estimates a class.

        This method implements a strategy for selecting one of these estimates
        """
        # print("Window: ", windowed_preds[0].shape)
        w_preds = torch.cat(windowed_preds, dim=1)  # batch, window, classes(18)
        # print("w_preds shape: ", w_preds.shape)
        #
        # batch, window (Recall that we scan the window of size max_seq_len across the sequence)
        values, class_indices = torch.max(w_preds, dim=2)  # batch, window
        if strategy == "max":
            indices = torch.argmax(values, dim=-1)  # batch
            # classes = class_indices[indices]
        else:  # strategy == "mode"
            most_common_class, indices = torch.mode(
                class_indices, dim=-1, keepdim=True
            )  # batch, 1
            # -----------------
            # print(
            #    "Class indices: ", class_indices, "  Most common class: ", most_common_class
            # )
            values = values * (class_indices == most_common_class).int()

            indices = torch.argmax(values, dim=-1)

        # -----------------
        # print(
        #    f"Windowed: shape={values.shape}  values={values}  indices={indices}"
        # )
        # print("True: ", label0.transpose(1, 0).shape)
        pred0 = w_preds[torch.arange(len(indices)), indices, :]

        return pred0
