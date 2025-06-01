import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import nltk
from nltk.tokenize import WordPunctTokenizer
import html

nltk.download('punkt')

# 超参数设置
BATCH_SIZE = 32
EMBEDDING_DIM = 128
EVENT_EMBED_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
EPOCHS = 10
MAX_LEN = 256
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

def clean_text(text):
    url_pattern = re.compile(r'http\S+')
    text = url_pattern.sub(r'', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# 使用NLTK的WordPunctTokenizer进行分词
def tokenize(text):
    return WordPunctTokenizer().tokenize(clean_text(text))

# 构建词汇表函数
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    # 添加特殊标记
    vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
    idx = 4
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

# 编码函数
def encode(text, vocab):
    tokens = tokenize(text)
    # 添加特殊标记：[CLS]开头，[SEP]结尾
    tokens = ['<CLS>'] + tokens + ['<SEP>']
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    
    # 截断或填充
    if len(ids) < MAX_LEN:
        ids += [vocab['<PAD>']] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 0, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

# 定义数据集类
class RumorDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df['text'].tolist()
        self.events = df['event'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        x_text = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        x_event = torch.tensor(self.events[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x_text, x_event, y

# 定义Transformer谣言检测模型
class TransformerRumorDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, num_events):
        super().__init__()
        
        # 文本嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, DROPOUT, MAX_LEN)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 事件嵌入层
        self.event_embedding = nn.Embedding(num_events, EVENT_EMBED_DIM)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + EVENT_EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1)
        )
        
        # 初始化权重
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
        # 创建注意力掩码（忽略填充位置）
        pad_mask = (text_input == 0)  # 假设0是填充索引
        
        # 文本嵌入 + 位置编码
        emb = self.embedding(text_input) * math.sqrt(EMBEDDING_DIM)
        emb = self.pos_encoder(emb)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(
            emb, 
            src_key_padding_mask=pad_mask
        )
        
        # 获取[CLS]标记的输出（位置0）作为整体表示
        cls_output = transformer_out[:, 0, :]
        
        # 事件嵌入
        event_features = self.event_embedding(event_input)
        
        # 拼接文本和事件特征
        combined = torch.cat((cls_output, event_features), dim=1)
        
        # 分类
        logits = self.classifier(combined)
        return logits.squeeze(1)

# 定义评估函数
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_text, x_event, y in loader:
            x_text, x_event, y = x_text.to(DEVICE), x_event.to(DEVICE), y.to(DEVICE)
            logits = model(x_text, x_event)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 定义主函数
def main():
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/val.csv')

    train_df['text'] = train_df['text'].fillna('')
    val_df['text'] = val_df['text'].fillna('')
    
    # 确保所有文本都是字符串类型
    train_df['text'] = train_df['text'].astype(str)
    val_df['text'] = val_df['text'].astype(str)

    vocab = build_vocab(train_df['text'])
    
    num_events = max(train_df['event'].max(), val_df['event'].max()) + 1
    
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    model = TransformerRumorDetector(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        num_events=num_events
    ).to(DEVICE)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # 训练模型
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x_text, x_event, y) in enumerate(train_loader):
            x_text, x_event, y = x_text.to(DEVICE), x_event.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x_text, x_event)
            loss = criterion(logits, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # 验证评估
        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'transformer_rumor_detector.pt')
            print(f'保存新的最佳模型，验证准确率: {val_acc:.4f}')
        
    print(f'训练完成，最佳验证准确率: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()