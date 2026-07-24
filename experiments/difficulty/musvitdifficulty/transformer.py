import torch
import torch.nn as nn

MAX_PAGES = 80

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=2, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(1, MAX_PAGES, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, pages):
        batch_size, seq_len, _ = x.size()

        if seq_len > MAX_PAGES:
            raise ValueError(
                f"Score has {seq_len} pages but the positional embedding only covers {MAX_PAGES}."
            )

        x = x + self.positional_embedding[:, :seq_len, :]
        mask = torch.arange(seq_len)[None, :].to(x.device) >= torch.tensor(pages)[:, None].to(x.device)

        x = self.transformer(x, src_key_padding_mask=mask)

        pooled = torch.stack([
            x[i, :pages[i]].mean(dim=0)
            for i in range(batch_size)
        ])

        return self.classifier(pooled)
