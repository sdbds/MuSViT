import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, mlp_dim=128):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, pages):
        packed = nn.utils.rnn.pack_padded_sequence(x, pages, batch_first = True, enforce_sorted = False)
        _, h_n = self.rnn(packed)
        return self.classifier(h_n[-1])
