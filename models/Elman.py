import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Elman Network (Simple Recurrent Neural Network) Model for time series forecasting.
    This model takes a sequence of input features and predicts a future sequence.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # Input feature dimension (number of series)
        self.c_out = configs.c_out    # Output feature dimension (usually 1 for univariate, or enc_in for multivariate)
        self.d_model = configs.d_model # Hidden dimension of Elman network
        self.e_layers = configs.e_layers # Number of Elman layers
        self.dropout = configs.dropout

        # Elman network (RNN with tanh activation by default)
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.elman = nn.RNN(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True,
            dropout=self.dropout if self.e_layers > 1 else 0 # Dropout only if more than one layer
        )

        # Linear layer to project RNN output to prediction length and output features
        # We take the output from the last time step and project it.
        self.projection = nn.Linear(self.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: (batch_size, seq_len, enc_in)
        # x_mark_enc: (batch_size, seq_len, features) - not used by simple Elman
        # x_dec: (batch_size, pred_len, dec_in) - not directly used by simple Elman for input
        # x_mark_dec: (batch_size, pred_len, features) - not used by simple Elman

        # Pass input through Elman network
        # output: (batch_size, seq_len, d_model)
        # hidden_state: (num_layers, batch_size, d_model)
        output, hidden_state = self.elman(x_enc)

        # Take the output from the last time step for prediction
        # output_last_timestep: (batch_size, d_model)
        output_last_timestep = output[:, -1, :]

        # Project the last time step's output to the desired prediction shape
        # projected_output: (batch_size, pred_len * c_out)
        projected_output = self.projection(output_last_timestep)

        # Reshape to (batch_size, pred_len, c_out)
        # This matches the expected output format of the forecasting framework
        final_output = projected_output.reshape(x_enc.size(0), self.pred_len, self.c_out)

        return final_output
