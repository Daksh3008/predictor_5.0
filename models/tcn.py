# models/tcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------
# TCN BLOCK: Dilated causal convolution + residual
# ----------------------------------------------------
class Chomp1d(nn.Module):
    """
    Removes extra padding on the right to maintain causality.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # match dimensions for residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


# ----------------------------------------------------
# FULL TCN MODEL
# ----------------------------------------------------
class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_levels=4, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        in_channels = input_dim

        # build multiple residual TCN blocks
        for i in range(num_levels):
            dilation = 2 ** i
            out_channels = hidden_dim
            layers.append(TemporalBlock(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        # final linear layer → predict 1-step smoothed return
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # → (batch, features, seq_len)
        y = self.tcn(x)
        y = y[:, :, -1]  # last step only → shape (batch, hidden)
        out = self.fc(y)
        return 0.05 * torch.tanh(out.squeeze(-1))
