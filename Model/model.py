import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    1D-CNN over the time axis (per-channel local patterns) feeding into an LSTM
    (temporal dynamics across the window), then a classification head.

    Input:  (batch, n_channels, n_samples)
    Output: (batch, 2) logits [interictal, preictal]
    """

    def __init__(self, n_channels, n_samples, cnn_channels=32, lstm_hidden=64, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels * 2, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_channels, n_samples)
        feat = self.conv(x)                 # (batch, cnn_channels*2, reduced_len)
        feat = feat.permute(0, 2, 1)         # (batch, seq_len, features) for LSTM
        lstm_out, _ = self.lstm(feat)        # (batch, seq_len, hidden*2)
        pooled = lstm_out.mean(dim=1)        # simple temporal average pooling
        return self.head(pooled)
