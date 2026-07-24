import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, mlp_dim=128):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, pages):
        pooled = torch.stack([
            x[i, :pages[i]].mean(dim=0)
            for i in range(x.size(0))
        ])

        return self.classifier(pooled)
