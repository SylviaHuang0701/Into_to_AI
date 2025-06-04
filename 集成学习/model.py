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
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
        Returns:
            带位置编码的张量 [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

class TransformerRumorDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, num_events=None, use_event=False):
        """
        Args:
            vocab_size: 词表大小
            embedding_dim: 词嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            ff_dim: 前馈网络维度
            num_events: 事件类型数量
            use_event: 是否使用事件信息
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.2, max_len=64)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_event = use_event

        # 分类器
        if self.use_event:
            self.event_embedding = nn.Embedding(num_events, 32)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim + 32, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )

        self.init_weights()

    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, text_input, event_input=None):
        """
        Args:
            text_input: 文本输入 [batch_size, seq_len]
            event_input: 事件输入 [batch_size] (可选)
        Returns:
            logits: 预测logits [batch_size]
        """
        # 处理文本输入
        pad_mask = (text_input == 0)
        emb = self.embedding(text_input) * torch.sqrt(torch.tensor(128.0))
        emb = self.pos_encoder(emb)
        transformer_out = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        cls_output = transformer_out[:, 0, :]  # 使用[CLS]标记的输出

        # 处理事件信息（如果使用）
        if self.use_event and event_input is not None:
            event_features = self.event_embedding(event_input)
            combined = torch.cat((cls_output, event_features), dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(cls_output)

        return logits.squeeze(1)