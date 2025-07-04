import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16)  # TODO implement "auto"

        self.embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_ff,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.mamba(enc_out)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len :, :]

        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)

        # other tasks not implemented
