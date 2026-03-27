import torch
import torch.nn as nn


"""
model.py
--------
Defines a lightweight bidirectional LSTM classifier for ASL recognition.

Input:
    X shape = (batch_size, T, 225)
Output:
    logits shape = (batch_size, num_classes)

This architecture is deliberately CPU-efficient and suitable for
sequence classification using MediaPipe keypoint vectors.
"""


class ASLLSTM(nn.Module):
    def __init__(
        self,
        input_dim=225,
        hidden_dim=256,
        num_layers=2,
        num_classes=100,
        dropout=0.3
    ):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Classification head
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, 225)

        Returns
        -------
        logits : torch.Tensor
            Shape (B, num_classes)
        """

        # LSTM forward
        out, _ = self.lstm(x)    # out shape: (B, T, hidden*2)

        # Use last time-step representation
        # Since sequences are padded, the padding is at the end,
        # so using out[:, -1, :] is valid.
        last_frame = out[:, -1, :]  # (B, hidden*2)

        logits = self.fc(last_frame)
        return logits