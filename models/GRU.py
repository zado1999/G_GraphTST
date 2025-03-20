import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(configs.d_model, configs.enc_in)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        x_enc, _ = self.gru(x_enc)
        enc_out = self.projection(x_enc)

        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        else:
            raise ValueError("Only forecast tasks implemented yet")
