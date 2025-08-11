import os
import math
import glob
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from scipy.spatial.transform import Rotation as R

from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, smallest_pow_2_greater_than, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

from typing_extensions import override
from typing import Literal, List, Union

from itertools import chain

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",
        seasonal_patterns="Yearly",
    ):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == "train":
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [
                v[~np.isnan(v)]
                for v in dataset.values[dataset.groups == self.seasonal_patterns]
            ]
        )  # split different frequencies
        self.ids = np.array(
            [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]]
        )
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[
            max(0, cut_point - self.seq_len) : cut_point
        ]
        insample[-len(insample_window) :, 0] = insample_window
        insample_mask[-len(insample_window) :, 0] = 1.0
        outsample_window = sampled_timeseries[
            max(0, cut_point - self.label_len) : min(
                len(sampled_timeseries), cut_point + self.pred_len
            )
        ]
        outsample[: len(outsample_window), 0] = outsample_window
        outsample_mask[: len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len :]
            insample[i, -len(ts) :] = ts_last_window
            insample_mask[i, -len(ts) :] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, "train.csv"))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = pd.read_csv(
            os.path.join(root_path, "test_label.csv")
        ).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "swat_train2.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "swat2.csv"))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(
        self,
        args,
        root_path,
        file_list=None,
        limit_size=None,
        flag=None,
        normalizer: Union[Normalizer, None] = None,
        label_encoder: Union[LabelEncoder, None] = None,
    ):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.label_encoder = label_encoder

        self.all_df, self.labels_df = self.load_all(
            root_path, file_list=file_list, flag=flag
        )
        # all sample IDs (integer indices 0 ... num_samples-1)
        self.all_IDs = self.all_df.index.unique()

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        if normalizer is not None:
            self.feature_df = normalizer.normalize(self.feature_df)

        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, "*"))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_path, "*"))
            )
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith(".ts")]
        if len(input_paths) == 0:
            pattern = "*.ts"
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(
            input_paths[0]
        )  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(
            filepath, return_separate_X_and_y=True, replace_missing_vals_with="NaN"
        )
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(
            labels.cat.codes, dtype=np.int8
        )  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)
        ).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if (
            np.sum(horiz_diffs) > 0
        ):  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if (
            np.sum(vert_diffs) > 0
        ):  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                .reset_index(drop=True)
                .set_index(pd.Series(lengths[row, 0] * [row]))
                for row in range(df.shape[0])
            ),
            axis=0,
        )

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if (
            self.root_path.count("EthanolConcentration") > 0
        ):  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(
                batch_x, labels, self.args
            )
            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class CMILoader(UEAloader):
    def __init__(
        self,
        args,
        root_path,
        data_path,
        limit_size=None,
        flag: Union[Literal["TRAIN", "TEST", "VALI"], None]=None,
        normalizer: Union[Normalizer, None] = None,
        label_encoder: Union[LabelEncoder, None] = None,
    ):
        super().__init__(
            args, root_path, [data_path], limit_size, flag, normalizer, label_encoder
        )

    @override
    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]] #.values
        labels = self.labels_df.loc[self.all_IDs[ind]].values

        if self.flag == "TRAIN":
            batch_x, _, _ = self.sample(batch_x, labels[0], ind, 1)

            batch_x = torch.from_numpy(batch_x[0].to_numpy())
        else:
            #batch_x = batch_x.values
            batch_x = torch.from_numpy(batch_x.to_numpy())

        labels = torch.from_numpy(labels)
        return batch_x.to(self.args.device), labels.to(self.args.device)

    def read_data(self, root_path, flag):
        print("Reading data")
        if flag in ["TRAIN", "VALI"]:
            df = pd.read_csv(f"{root_path}/train.csv")
            demo = pd.read_csv(f"{root_path}/train_demographics.csv")

            df = pd.merge(df, demo, on="subject")

            assert (
                not self.label_encoder is None
            ), "LabelEncoder should be specified for training and validation."

            mask = df["sequence_type"] == "Non-Target"
            df.loc[mask, "gesture"] = "Non-Target"
            if self.args.is_binary:
                mask = df["sequence_type"] == "Target"
                df.loc[mask, "gesture"] == "Target"

            if flag == "TRAIN":
                df["gesture_int"] = self.label_encoder.fit_transform(df["gesture"])
            else:
                df["gesture_int"] = self.label_encoder.transform(df["gesture"])

        else:
            df = pd.read_csv(f"{root_path}/test.csv")
            
            df["gesture_int"] = self.label_encoder.transform(df["gesture"])

        return df
    

    def relevant_features(self, columns:List[str]):
        # Feature list
        imu_cols = [
            c for c in columns if (c.startswith("rot_") or c.startswith("acc_"))
        ]
        tof_cols = [
            c for c in columns if c.startswith("thm_") or c.startswith("tof_")
        ]

        if self.args.use_imu_only:
            feature_cols = imu_cols

            if self.args.use_acceleration_only:
                feature_cols = ["acc_x", "acc_y", "acc_z"]
        else:
            feature_cols = imu_cols + tof_cols

        feature_cols.extend(["handedness", "sequence_counter"])

        return feature_cols


    @override
    def load_all(self, root_path, file_list=None, flag=None):
        """
        The differences between this and the parent are that:
        - The data lives in a single csv file,
        """
        df = self.read_data(root_path, flag)

        df = df[df["behavior"] == "Performs gesture"]
        df = df.dropna()

        # Feature list
        feature_cols = self.relevant_features(df.columns)

        self.args.enc_in = len(feature_cols)
        self.args.d_model = self.args.multiplier * smallest_pow_2_greater_than(self.args.enc_in)
        self.args.d_ff = 2 * self.args.d_model

        new_columns = ["sequence_id", "gesture_int"]
        new_columns.extend(feature_cols)
        df = df[new_columns]

        # new_columns = ["index", "time"]
        new_columns = ["index"]
        new_columns.extend(feature_cols)
        print("Columns: ", new_columns)
        dtypes = {feature: "float" for feature in feature_cols}
        dtypes["index"] = "int64"
        # dtypes["time"] = "float"

        all_df = pd.DataFrame(columns=new_columns)  # , dtype=dtypes)
        all_df = all_df.astype(dtypes)

        max_seq_len, min_seq_len, Xf, labels = self.normalize_seq_len(
            df, feature_cols, flag
        )

        #self.args.c_out = len(df["gesture_int"].unique())
        self.max_seq_len = max_seq_len
        print("Max seq lenght: ", self.max_seq_len)
        print("Min seq length: ", min_seq_len)

        all_df = pd.concat(Xf, axis=0)
        print(all_df.columns)

        print(len(Xf), len(labels))

        if "gesture_int" in all_df.columns:
            all_df.drop(columns=["gesture_int"], inplace=True)
        if "sequence_id" in all_df.columns:
            all_df.drop(columns=["sequence_id"], inplace=True)

        print("Data shape: ", all_df.shape)
        print("Head\n", all_df.head(10))
        print("Tail\n", all_df.tail(10))
        try:
            print("Middle\n", all_df[100])
            print("Middle\n", all_df[200])
            print("Middle\n", all_df[400])
        except Exception as exp:
            print("Some of these keys were not found!")
        
        all_df.drop(columns=["sequence_counter"], inplace=True)

        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(
            labels.cat.codes, dtype=np.int8
        )  # int8-32 gives an error when using nn.CrossEntropyLoss

        # Replace NaN values
        grp = all_df.groupby(by=all_df.index)
        print("Groups: ", len(grp))
        #all_df = grp.transform(interpolate_missing)

        return all_df, labels_df


    def expanding_window_scan(self, index: int, zdf: pd.DataFrame, label: int, min_win_sz: int, max_win_sz: int):
        """"
            Generate additional data by scanning the given dataframe using an expanding window.

            zdf: The dataframe on which to scan the expanding window,
            min_len: The minimum size of the window. It is some percentate of the max_seq_len.
            max_len: The maximum size of the window. It is min(len(zdf), max_seq_len)
        """
        Xf = []
        labels = []

        for end in chain([len(zdf)], range(min_win_sz, max_win_sz)):
            z = zdf.iloc[:end]
            z = z.copy(deep=True)

            xs, ls, index = self.sample(z, label, index, 0)
            
            Xf.extend(xs)
            labels.extend(ls)

            #-----------------------------------------------------------------
            idx = np.ones((z.shape[0],)) * index
            z.set_index(pd.Index(idx), inplace=True)

            Xf.append(z)
            labels.append(label)

            index = index + 1
            #-----------------------------------------------------------------

            if index % 50000 == 0:
                print(index)

        return Xf, labels, index


    def normalize_seq_len(
        self,
        # Contains data about the behaviour of interest and is grouped by sequence_id
        df: pd.DataFrame,
        feature_cols: List[str],
        flag: str,
    ):
        labels = []
        X_list: List[pd.DataFrame] = []
        lens: List[int] = []
        seq_ids: List[int] = []

        rot_as_mat_headers = [f"rot-{i}" for i in range(9)]

        index = 0
        seq_gp = df.groupby("sequence_id")
        for seq_id, seq in seq_gp:

            # Pre-conditions:
            # test_seq_ids -> Empty when reading training and test data,
            #              -> Contains seq ids when reading validation data.
            if seq_id not in self.args.test_seq_ids:
                labels.append(seq["gesture_int"].values[0])
                seq_ids.append(seq_id)

                seq = seq[feature_cols]
                seq = seq.copy()
                idx = np.ones((seq.shape[0],)) * index
                seq.set_index(pd.Index(idx), inplace=True)

                idr = np.argmax(np.array(["rot_" in header for header in seq.columns]))
                ids = idr + 4
                assert np.all(["rot_" in header for header in seq.columns[idr:ids]]) == True, "The following operation should be applied to acceleration data."

                #rots_as_mat = [R.from_quat(np.flip(s[idr:ids])).as_matrix().reshape((1, -1)).squeeze() for s in seq.to_numpy().astype(np.float64)]
                rots_as_mat = [R.from_quat(np.flip(s[idr:ids])).as_matrix().reshape((1, -1)).squeeze() for s in seq.to_numpy().astype(np.float64)]
                seq[rot_as_mat_headers] = np.array(rots_as_mat)
                
                index = index + 1

                X_list.append(seq)
                lens.append(len(seq))

        if flag in ["VALI", "TEST"]:  # Validation dataset
            max_seq_len = np.max(lens)
            min_seq_len = np.min(lens)

            return max_seq_len, min_seq_len, X_list, labels
        else: # flag == "TRAIN":
            # Split the data into training and validation sets.
            ids = list(range(len(X_list)))

            # Split
            ids_tr, ids_val, y_tr, y_val = train_test_split(
                ids, labels, test_size=0.2, random_state=self.args.seed, stratify=labels
            )

            X_list = [X_list[id] for id in ids_tr]
            labels = y_tr

            max_seq_len = int(np.percentile(lens, self.args.pad_percentile))
            min_seq_len = np.min(lens)
            print("Max seq length percentile: ", self.args.pad_percentile, max_seq_len)
            lens = [lens[id] for id in ids_tr]
            print("Max seq length percentile: ", self.args.pad_percentile, int(np.percentile(lens, self.args.pad_percentile)))

            print("Len features vs labels: ", len(X_list), len(labels))
            print("Split: ", len(ids), len(ids_tr))

            self.args.test_seq_ids = {seq_ids[id] for id in ids_tr}

        min_len = int(0.5 * max_seq_len)

        new_labels: List[int] = []
        Xf = []

        index = 0
        for seq, label, length in zip(X_list, labels, lens):

            if True or length >= min_len:
                # We only need part of the sequence to determine if it is interesting
                # Use this observation to generate new data
                # This simulates a decoder only transformer that can ingest one token at a time.
                # Actually, this approach slows down learning and inference :-(
                z = seq
                output: List[torch.Tensor] = []
                if length > max_seq_len:
                    num_features = len(z.columns)
                    # unfold only works with 4D tensors
                    # Therefore the 2D tensor is unsqueezed twice to add two more dimensions.
                    zt = torch.tensor(z.to_numpy(copy=False)).unsqueeze(0).unsqueeze(0)
                    out_tensor = F.unfold(zt, kernel_size=(max_seq_len, num_features), stride=(1, 1))

                    output = [out.reshape(-1, num_features) for out in out_tensor.squeeze().transpose(1, 0)]
                else:
                    output = [torch.tensor(z.values)]

                for zt in output:
                    zdf = pd.DataFrame(zt.cpu().numpy(), columns=seq.columns)
                    length = min(len(zt), max_seq_len)

                    Xf_, labels_, index = self.expanding_window_scan(index, zdf, label, min_len, length) 

                    Xf.extend(Xf_)
                    new_labels.extend(labels_)

        return max_seq_len, min_seq_len, Xf, new_labels

    
    def sample(self, seq: pd.DataFrame, label:int, index:int, N:int=3):
        """
            N: Number of measurement samples
        """
        min_acc = -0.30/2
        max_acc =  0.30/2

        min_rot = -3.50/2 # Degrees
        max_rot =  3.50/2 # Degrees

        ida = np.argmax(np.array(["acc" in header for header in seq.columns]))
        idb = ida + 3
        assert np.all(["acc" in header for header in seq.columns[ida:idb]]) == True, "The following operation should be applied to acceleration data."
        acc_headers = [h for h in seq.columns if "acc" in h]

        idr = np.argmax(np.array(["rot" in header for header in seq.columns]))
        ids = idr + 4
        assert np.all(["rot_" in header for header in seq.columns[idr:ids]]) == True, "The following operation should be applied to acceleration data."
        rot_headers = [h for h in seq.columns if "rot_" in h]

        rot_as_mat_headers = [f"rot-{i}" for i in range(9)]

        Xf: List[pd.DataFrame] = []
        labels: List[int] = []
        for _ in range(N):
            # Sample from acceleration distribution: A uniform distribution between -0.30 to 0.30
            acc_noise = np.random.uniform(low=min_acc, high=max_acc, size=(len(seq), 3))

            # Sample Euler angles then convert them to quaternions.
            rot_noise = np.random.uniform(low=min_rot, high=max_rot, size=(len(seq), 3))
            rot_noise = R.from_euler("xyz", rot_noise, degrees=True)
            #rot_noise = rot_noise.as_quat(scalar_first=True) # w comes first: w, x, y, z

            tseq = seq.copy(deep=True)
            ms = tseq.to_numpy(copy=False)

            #acc  = ms[:, ida:idb] + acc_noise
            tseq[acc_headers] = tseq[acc_headers].to_numpy() + acc_noise
            #rot_mat = R.from_quat(ms[:, idr:ids]).as_matrix() @ rot_noise.as_matrix() 
            rot_mat = R.from_quat(tseq[rot_headers].to_numpy()).as_matrix() @ rot_noise.as_matrix() 
            rot = R.from_matrix(rot_mat).as_quat() # x, y, z, w
            rot = np.fliplr(rot) # w, x, y, z

            #ms[:, ida:idb] = acc
            #ms[:, idr:ids] = rot
            tseq[rot_headers] = rot

            #rots_as_mat = [R.from_quat(np.flip(s[idr:ids])).as_matrix().reshape((1, -1)).squeeze() for s in tseq.to_numpy().astype(np.float64)]
            #tseq[rot_as_mat_headers] = np.array(rots_as_mat)
            tseq[rot_as_mat_headers] = rot_mat.reshape((tseq.shape[0], -1))

            idx = np.ones((tseq.shape[0],)) * index
            tseq.set_index(pd.Index(idx), inplace=True)

            Xf.append(tseq)
            labels.append(label)

            index = index + 1

        #idx = np.ones((seq.shape[0],)) * index
        #seq.set_index(pd.Index(idx), inplace=True)

        #Xf.append(seq)
        #labels.append(label)

        #index = index + 1

        return Xf, labels, index
            


