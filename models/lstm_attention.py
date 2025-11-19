# models/lstm_attention.py
import torch
import torch.nn as nn

class LSTMAttn(nn.Module):
    def __init__(self, input_dim, lstm_hidden=192, lstm_layers=3, dropout=0.3, attn_heads=2, attn_dim=128, mlp_hidden=128, bidirectional=False):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # project LSTM hidden dim to attention embed dim
        self.attn_proj = nn.Linear(lstm_hidden * self.num_directions, attn_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(attn_dim)
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)
        proj = self.attn_proj(lstm_out)  # (batch, seq_len, attn_dim)
        attn_out, _ = self.attn(proj, proj, proj)  # self-attention
        out = self.layernorm(attn_out + proj)
        last = out[:, -1, :]  # use last timestep aggregated embedding
        out = self.mlp(last)
        return 0.05 * torch.tanh(out.squeeze(-1))  # (batch,)

    @torch.no_grad()
    def predict_returns(self, seq_batch):
        """seq_batch: tensor (batch, seq_len, features)"""
        self.eval()
        device = next(self.parameters()).device
        xb = seq_batch.to(device)
        out = self.forward(xb)
        return out.cpu().numpy()
