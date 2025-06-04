import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re
import json
import model
import nltk
from nltk.tokenize import WordPunctTokenizer
import html

nltk.download('punkt')

# 超参数设置
BATCH_SIZE = 32
EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
EPOCHS = 10
MAX_LEN = 128
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def clean_text(text):
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'#(\w+)', r'HASHTAG\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = html.unescape(text)
    text = re.sub(r'http\S+', 'URL', text)
    return text.lower().strip()

def tokenize(text):
    return WordPunctTokenizer().tokenize(clean_text(text))

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
    idx = 4
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    tokens = ['<CLS>'] + tokens + ['<SEP>']
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab['<PAD>']] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

class RumorDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x_text = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x_text, y

def evaluate(transformer_model, loader):
    transformer_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_text, y in loader:
            x_text, y = x_text.to(DEVICE), y.to(DEVICE)
            logits = transformer_model(x_text)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def main():
    torch.manual_seed(2025)
    train_df = pd.read_csv('../data/train_new.csv')
    val_df = pd.read_csv('../data/val.csv')
    train_df['text'] = train_df['text'].fillna('')
    val_df['text'] = val_df['text'].fillna('')
    vocab = build_vocab(train_df['text'])
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    transformer_model = model.TransformerRumorDetector(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM
    ).to(DEVICE)
    optimizer = optim.AdamW(transformer_model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    warmup_iters = min(500, len(train_loader))
    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, 0.1)
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        transformer_model.train()
        total_loss = 0
        for batch_idx, (x_text, y) in enumerate(train_loader):
            x_text, y = x_text.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = transformer_model(x_text)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(transformer_model.parameters(), 1.0)
            optimizer.step()
            if epoch == 0 and batch_idx < warmup_iters:
                warmup_scheduler.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        val_acc = evaluate(transformer_model, val_loader)
        avg_loss = total_loss / len(train_loader)
        scheduler.step(val_acc)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'transformer_rumor_detector2.pt')
            print(f'保存新的最佳模型，验证准确率: {val_acc:.4f}')
    print(f'训练完成，最佳验证准确率: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()
