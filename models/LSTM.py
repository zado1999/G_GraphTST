import torch
import torch.nn as nn

class Model(nn.Module):
    """
    LSTM Model for time series forecasting.
    This model takes a sequence of input features and predicts a future sequence.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # Input feature dimension (number of series)
        self.c_out = configs.c_out    # Output feature dimension (usually 1 for univariate, or enc_in for multivariate)
        self.d_model = configs.d_model # Hidden dimension of LSTM
        self.e_layers = configs.e_layers # Number of LSTM layers
        self.dropout = configs.dropout

        # LSTM layer(s)
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True,
            dropout=self.dropout if self.e_layers > 1 else 0 # Dropout only if more than one layer
        )

        # Linear layer to project LSTM output to prediction length and output features
        # The output of LSTM is (batch, seq_len, d_model) for each time step.
        # We are interested in the last hidden state for forecasting.
        # However, for sequence-to-sequence prediction, we might need to process all outputs.
        # For simplicity, let's assume we take the output of the last time step
        # and project it to pred_len * c_out.
        # A more robust seq2seq LSTM would involve an encoder-decoder structure.
        # Here, we'll use a simple linear projection from the d_model output of the last time step
        # to the desired output shape (pred_len * c_out).
        self.projection = nn.Linear(self.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: (batch_size, seq_len, enc_in)
        # x_mark_enc: (batch_size, seq_len, features) - not used by simple LSTM
        # x_dec: (batch_size, pred_len, dec_in) - not directly used by simple LSTM for input
        # x_mark_dec: (batch_size, pred_len, features) - not used by simple LSTM

        # Pass input through LSTM
        # output: (batch_size, seq_len, d_model)
        # hidden_state: (num_layers, batch_size, d_model)
        # cell_state: (num_layers, batch_size, d_model)
        output, (hidden_state, cell_state) = self.lstm(x_enc)

        # Take the output from the last time step for prediction
        # Alternatively, could use the last hidden state: hidden_state[-1]
        # output_last_timestep: (batch_size, d_model)
        output_last_timestep = output[:, -1, :]

        # Project the last time step's output to the desired prediction shape
        # projected_output: (batch_size, pred_len * c_out)
        projected_output = self.projection(output_last_timestep)

        # Reshape to (batch_size, pred_len, c_out)
        # This matches the expected output format of the forecasting framework
        final_output = projected_output.reshape(x_enc.size(0), self.pred_len, self.c_out)

        return final_output
