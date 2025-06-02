import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 0, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

class TransformerRumorDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, num_events):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, 0.1, 256)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.event_embedding = nn.Embedding(num_events, 32)
        self.event_pos_encoder = PositionalEncoding(32, 0.1, 1)
        combined_d_model = embedding_dim + 32
        combined_encoder_layer = TransformerEncoderLayer(
            d_model=combined_d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.combined_transformer_encoder = TransformerEncoder(combined_encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(combined_d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, text_input, event_input):
        pad_mask = (text_input == 0)
        emb = self.embedding(text_input) * torch.sqrt(torch.tensor(128.0))
        emb = self.pos_encoder(emb)
        transformer_out = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        cls_output = transformer_out[:, 0, :]
        event_features = self.event_embedding(event_input)
        combined = torch.cat((cls_output, event_features), dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(1)