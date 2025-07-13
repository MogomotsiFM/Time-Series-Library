import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_cmi_classification import Exp_CMI_Classification
from utils.print_args import print_args
import random
import numpy as np

from types import SimpleNamespace

from sklearn.preprocessing import LabelEncoder

from data_provider.uea import Normalizer


def wrapper(args):
    if torch.cuda.is_available() and args["use_gpu"]:
        args["device"] = torch.device("cuda:{}".format(args["gpu"]))
        print("Using GPU")
    else:
        if hasattr(torch.backends, "mps"):
            args["device"] = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            args["device"] = torch.device("cpu")
        print("Using cpu or mps")

    if args["use_gpu"] and args["use_multi_gpu"]:
        args["devices"] = args["devices"].replace(" ", "")
        device_ids = args["devices"].split(",")
        args["device_ids"] = [int(id_) for id_ in device_ids]
        args["gpu"] = args["device_ids"][0]

    print("Args in experiment:")
    print(args)

    # args = SimpleNamespace(**args)

    if args["task_name"] == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args["task_name"] == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args["task_name"] == "imputation":
        Exp = Exp_Imputation
    elif args["task_name"] == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args["task_name"] == "classification":
        # Exp = Exp_Classification
        Exp = Exp_CMI_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    normalizer = Normalizer()
    label_encoder = LabelEncoder()

    if args["is_training"]:
        for ii in range(args["itr"]):
            # setting record of experiments
            # args = SimpleNamespace(**args)
            exp = Exp(SimpleNamespace(**args), normalizer, label_encoder)
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
                args["task_name"],
                args["model_id"],
                args["model"],
                args["data"],
                args["features"],
                args["seq_len"],
                args["label_len"],
                args["pred_len"],
                args["d_model"],
                args["n_heads"],
                args["e_layers"],
                args["d_layers"],
                args["d_ff"],
                args["expand"],
                args["d_conv"],
                args["factor"],
                args["embed"],
                args["distil"],
                args["des"],
                ii,
            )

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            if args["gpu_type"] == "mps":
                torch.backends.mps.empty_cache()
            elif args["gpu_type"] == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args["task_name"],
            args["model_id"],
            args["model"],
            args["data"],
            args["features"],
            args["seq_len"],
            args["label_len"],
            args["pred_len"],
            args["d_model"],
            args["n_heads"],
            args["e_layers"],
            args["d_layers"],
            args["d_ff"],
            args["expand"],
            args["d_conv"],
            args["factor"],
            args["embed"],
            args["distil"],
            args["des"],
            ii,
        )

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        if args["gpu_type"] == "mps":
            torch.backends.mps.empty_cache()
        elif args["gpu_type"] == "cuda":
            torch.cuda.empty_cache()


def intro():
    config = dict()

    # Original code parameters
    config["optimizer"] = optimizer
    config["strategy"] = strategy
    config["use_imu_only"] = True
    config["use_acceleration_only"] = True
    config["pad_percentile"] = pad_percentile
    config["drop_last"] = True
    config["top_k"] = 4  # TimesNet parameter

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    config["seed"] = fix_seed

    # parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    config["task_name"] = "classification"
    config["is_training"] = 1
    config["model_id"] = "test"
    config["model"] = model_name
    # parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
    #                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # parser.add_argument('--model', type=str, required=True, default='Autoformer',
    #                    help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    config["data"] = "CMI"
    config["root_path"] = (
        "C:\\Users\\GAME\\Desktop\\Projects\\kaggle\\cmi\\cmi-detect-behavior-with-sensor-data"
    )
    config["data_path"] = "train.csv"
    config["features"] = "MS"
    config["target"] = "OT"  # ???
    config["freq"] = "s"
    config["timeenc"] = 1  # 0 or 1. Used in the data_provider/data_loader.py
    config["checkpoints"] = "./checkpoints"

    # parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    # parser.add_argument('--features', type=str, default='M',
    #                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    # parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # parser.add_argument('--freq', type=str, default='h',
    #                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    config["seq_len"] = seq_len
    config["label_len"] = 1
    config["pred_len"] = 0
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--label_len', type=int, default=48, help='start token length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    # parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    # parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    config["expand"] = 2
    config["d_conv"] = 4
    # parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    # parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    # parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    config["enc_in"] = d_model
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')

    config["c_out"] = num_classes
    # parser.add_argument('--c_out', type=int, default=7, help='output size')

    config["d_model"] = 30 * d_model
    # parser.add_argument('--d_model', type=int, default=512, help='dimension of model')

    config["n_heads"] = h
    # parser.add_argument('--n_heads', type=int, default=8, help='num of heads')

    config["e_layers"] = N
    # parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')

    config["d_layers"] = 0
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    config["d_ff"] = d_ff
    # parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

    # Used in TimeMixer model
    config["moving_avg"] = 3
    # parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

    config["factor"] = 1
    # parser.add_argument('--factor', type=int, default=1, help='attn factor')

    config["distil"] = True
    # parser.add_argument('--distil', action='store_false',
    #                    help='whether to use distilling in encoder, using this argument means not using distilling',
    #                   default=True)

    config["dropout"] = dropout
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    config["embed"] = "timeF"
    # parser.add_argument('--embed', type=str, default='timeF',
    #                    help='time features encoding, options:[timeF, fixed, learned]')

    config["activation"] = "gelu"
    # parser.add_argument('--activation', type=str, default='gelu', help='activation')

    config["channel_independence"] = 0
    # parser.add_argument('--channel_independence', type=int, default=1,
    #                    help='0: channel dependence 1: channel independence for FreTS and TimeMixer models')

    config["decomp_method"] = "dft_decomp" # Options: dft_decomp, moving_avg
    # parser.add_argument('--decomp_method', type=str, default='moving_avg',
    #                    help='method of series decompsition, only support moving_avg or dft_decomp')

    config["use_norm"] = 0
    # Used in the TimeMixer model to determine whether to normalize the data or not. If the data is already normalized then
    # this can be set to False
    # Actually, normalization is not used for classification
    # parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # Used in TimeMixer model
    config["down_sampling_layers"] = 1
    # parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')

    # Used in TimeMixer model
    config["down_sampling_window"] = 1
    config["down_sampling_method"] = "conv"
    # parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    # parser.add_argument('--down_sampling_method', type=str, default=None,
    #                    help='down sampling method, only support avg, max, conv')
    # parser.add_argument('--seg_len', type=int, default=96,
    #                    help='the length of segmen-wise iteration of SegRNN')

    # optimization
    config["num_workers"] = 2
    config["itr"] = 1
    config["train_epochs"] = train_epochs
    config["batch_size"] = batch_size
    config["patience"] = 100
    config["learning_rate"] = 0.00001
    config["des"] = "CMI"
    config["loss"] = "CrossEntropy"
    # Strategies to adjust the learning rate: type1, type2, type3, consine
    config["lradj"] = "type3"
    config["use_amp"] = False
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # parser.add_argument('--itr', type=int, default=1, help='experiments times')
    # parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--des', type=str, default='test', help='exp description')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    config["use_gpu"] = False
    config["gpu"] = 0
    config["gpu_type"] = "cuda"
    config["use_multi_gpu"] = False
    config["devices"] = "0"
    # parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    config["p_hidden_dims"] = [256]  # [128, 128]
    config["p_hidden_layers"] = len(config["p_hidden_dims"])  # 2
    # parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
    #                    help='hidden layer dimensions of projector (List)')
    # parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    config["use_dtw"] = False
    # parser.add_argument('--use_dtw', type=bool, default=False,
    #                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    config["augmentation_ratio"] = 0
    config["seed"] = 42
    # parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    # parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    # parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")

    config["scaling"] = True
    config["permutation"] = False
    config["randompermutation"] = False
    config["magwarp"] = False
    config["timewarp"] = False
    # parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    # parser.add_argument('--permutation', default=False, action="store_true",
    #                    help="Equal Length Permutation preset augmentation")
    # parser.add_argument('--randompermutation', default=False, action="store_true",
    #                    help="Random Length Permutation preset augmentation")
    # parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    # parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")

    config["windowslice"] = False
    config["windowwarp"] = False
    config["rotation"] = False
    config["spawner"] = False
    config["dtwwarp"] = True
    config["shapedtwwarp"] = False
    config["wdba"] = False
    config["discdtw"] = False
    config["discsdtw"] = False
    config["extra_tag"] = ""
    # parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    # parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    # parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    # parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    # parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    # parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    # parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    # parser.add_argument('--discdtw', default=False, action="store_true",
    #                    help="Discrimitive DTW warp preset augmentation")
    # parser.add_argument('--discsdtw', default=False, action="store_true",
    #                    help="Discrimitive shapeDTW warp preset augmentation")
    # parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    # parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    # args = parser.parse_args()

    wrapper(config)


if __name__ == "__main__":
    # model name, options: [Autoformer, Transformer, TimesNet, TimeMixer, Mamba, TemporalFusionTransformer]
    model_name = "TimeMixer"
    num_classes = 18
    seq_len = 35
    pad_percentile = 0.95
    d_model = 3
    N: int = 8
    h: int = 1
    dropout: float = 0.1
    d_ff: int = 256
    device = None
    train_epochs = 500
    batch_size = 32
    fix_seed = 1983
    strategy = "max"  # max, mode, all, sum
    optimizer = "adam"  # adam, sgd

    intro()
