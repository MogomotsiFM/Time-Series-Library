from exp.exp_classification import Exp_Classification
from data_provider.data_loader import CMILoader

from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import LabelEncoder

from functools import partial
from typing import Union
from types import SimpleNamespace

from data_provider.uea import collate_fn, smallest_pow_2_greater_than, Normalizer
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
from typing_extensions import override, overload

warnings.filterwarnings("ignore")


class Exp_CMI_Classification(Exp_Classification):
    def __init__(
        self,
        args,
        normalizer: Union[Normalizer, None] = None,
        label_encoder: Union[LabelEncoder, None] = None,
    ):
        super().__init__(args, normalizer, label_encoder)

    @override
    def _build_model(self):
        setting = Exp_CMI_Classification.format_settings(self.args)

        # normalizer = Normalizer()
        # label_encoder = LabelEncoder()
        # We use this to record the list of sequence ids that are used for training.
        # The rest of the sequence ids are used for validation.
        self.args.test_seq_ids = set()

        # model input depends on data
        self.train_data, self.train_loader = self._get_data(
            flag="TRAIN", normalizer=self.normalizer, label_encoder=self.label_encoder
        )
        self.vali_data, self.vali_loader = self._get_data(
            flag="VALI",
            max_seq_len=self.train_data.max_seq_len,
            normalizer=self.normalizer,
            label_encoder=self.label_encoder,
        )

        self.args.max_seq_len = self.train_data.max_seq_len
        self.args.seq_len = self.train_data.max_seq_len

        self.args.pred_len = 0
        self.args.enc_in = self.train_data.feature_df.shape[1]
        self.args.num_class = len(self.train_data.class_names)

        projection_dim = smallest_pow_2_greater_than(self.args.d_model * self.args.max_seq_len)
        self.args.p_hidden_dims = [
            projection_dim,
            (
                projection_dim // 2
                if projection_dim > (2 * self.args.c_out)
                else projection_dim
            ),
        ]
        self.args.p_hidden_layers = len(self.args.p_hidden_dims)

        self.args.test_seq_ids = set()
        print("Model parameters: ", self.args)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        try:
            print("\nTry loading model parameters")
            file_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            print(file_path)
            model.load_state_dict(
                torch.load(file_path, map_location=self.args.device)
            )
            print("\nSuccessfully loaded pre-existing model parameters.")
        except Exception as exp:
            print(exp)
            print("\nModel parameters not found.")
            print("Training a model from scratch...")

        return model

    # @override
    # def _get_data(self, flag):
    #    return super()._get_data(flag)

    @staticmethod
    def format_settings(args):
        settings = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            0,
        )

        return settings


    # @overload
    def _get_data(
        self,
        flag,
        max_seq_len=0,
        normalizer: Union[Normalizer, None] = None,
        label_encoder: Union[LabelEncoder, None] = None,
    ):
        print("Data loader: ", type(self))

        shuffle_flag = False if (flag == "vali" or flag == "VALI") else True
        # batch_size = 1 if (flag == "vali" or flag == "VALI") else self.args.batch_size
        batch_size = self.args.batch_size  if flag == "TRAIN" else 2*self.args.batch_size

        data_set = CMILoader(
            args=self.args,
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            # limit_size=100 * self.args.batch_size,
            flag=flag,
            normalizer=normalizer,
            label_encoder=label_encoder,
        )

        max_seq_len = max(max_seq_len, data_set.max_seq_len)
        print(type(self), f"Max seq len-{flag}: ", max_seq_len)
        print("Device: ", self.args.device, self.device)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=True,
            generator=torch.Generator(device=self.args.device),
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            collate_fn=partial(collate_fn, max_len=max_seq_len),
        )
        return data_set, data_loader

    @override
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
                masks = []
                for end in range(self.args.max_seq_len, vali_data.max_seq_len):
                    start = end - self.args.max_seq_len

                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label0.to(self.device)

                    b = batch_x[:, start:end, :]
                    m = padding_mask[:, start:end]  # batch, max_seq_len

                    # This ensures that each sequence in a batch is used at least once
                    # A better solution is to use the subsequence that yields the highest score
                    seq_len = torch.sum(m, dim=1)  # batch, seq_len => batch
                    #seq_len = torch.sum(m, dim=1, keepdim=True)  # batch, seq_len => batch
                    min_seq_len_mask = torch.greater_equal(
                        seq_len, 0.5 * self.args.max_seq_len
                    )

                    min_seq_len_mask = torch.logical_or(
                        min_seq_len_mask, torch.tensor([start]) == 0
                    )
                    print(f"\r               \r {i}  {start}", end="")

                    if not torch.any(min_seq_len_mask):
                        break

                    outputs = self.model(b, m, None, None)

                    windowed_preds.append(
                        torch.softmax(
                            torch.reshape(outputs, (-1, 1, self.args.c_out)),
                            dim=-1,
                        )
                    )

                    masks.append(min_seq_len_mask)

                    outputs = outputs[min_seq_len_mask]

                    pred = outputs.detach().cpu()

                    label = label[min_seq_len_mask]

                    loss = criterion(pred, label.long().squeeze(-1).cpu())
                    total_loss.append(loss)

                    preds.append(outputs.detach())
                    trues.append(label)

                pred0, label0 = Exp_CMI_Classification.select_best_predictions(
                    windowed_preds, label0, masks, strategy=self.args.strategy
                )
                loss0 = criterion(pred0.cpu(), label0.long().squeeze(-1).cpu())
                total_win_loss.append(loss0)

                all_win_preds.append(pred0)
                all_win_trues.append(label0)

        total_loss = np.average(total_loss)
        total_win_loss = np.average(total_win_loss)

        all_win_preds = torch.cat(all_win_preds, dim=0)
        all_win_trues = torch.cat(all_win_trues, dim=0)
        predictions = torch.argmax(all_win_preds, dim=1).cpu().numpy()
        all_win_trues = all_win_trues.flatten().cpu().numpy()
        win_accuracy = cal_accuracy(predictions, all_win_trues)

        print("\n", len(preds), len(trues))
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
        windowed_preds: List[torch.Tensor],
        labels: torch.Tensor,
        masks: List[torch.Tensor],
        strategy: Literal["all", "max", "mode", "sum"] = "mode",
    ):
        """
        For validation purposes
        During validation we scan a window of size max_seq_len across each of the samples and pass it through
        the model. Each of these windows estimates a class.

        This method implements a strategy for selecting one of these estimates
        """
        masks = [m.view((-1, 1)) for m in masks]
        w_preds = torch.cat(windowed_preds, dim=1)  # batch, window, classes(18)
        values, class_indices = torch.max(w_preds, dim=2)  # batch, window
        if strategy == "max":
            indices = torch.argmax(values, dim=-1)  # batch

            pred0 = w_preds[torch.arange(len(indices)), indices, :]

            mask = torch.cat(masks, dim=-1)
            mask = mask[torch.arange(len(indices)), indices]
            
            pred0 = pred0[mask]
            labels = labels[mask]
            
            return pred0, labels

        elif strategy == "mode":
            most_common_class, indices = torch.mode(
                class_indices, dim=-1, keepdim=True
            )  # batch, 1
            values = torch.masked_fill(
                values,
                torch.logical_not(class_indices == most_common_class),
                torch.tensor(float("-inf")),
            )

            indices = torch.argmax(values, dim=-1)

            pred0 = w_preds[torch.arange(len(indices)), indices, :]

            mask = torch.cat(masks, dim=-1)
            mask = mask[torch.arange(len(indices)), indices]
            
            pred0 = pred0[mask]
            labels = labels[mask]

            return pred0, labels

        elif strategy == "sum":
            w_preds = torch.sum(w_preds, dim=1)  # batch, classes

            pred0 = torch.softmax(w_preds, dim=-1)

            return pred0, labels

        else:  # Use all the predictions
            pred0 = torch.reshape(w_preds, (-1, w_preds.shape[-1]))

            labels = labels.reshape((-1, 1)).repeat_interleave(
                repeats=w_preds.shape[1], dim=0
            )

            mask = torch.cat(masks, dim=0)

            mask = torch.squeeze(mask, dim=-1)

            pred0 = pred0[mask]
            labels = labels[mask]

            return pred0, labels
