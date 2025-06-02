import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re
import numpy as np
import model  # 导入模型模块
import os
import time
import html
import nltk
from nltk.tokenize import WordPunctTokenizer
import json
nltk.download('punkt')

BATCH_SIZE = 32
EMBEDDING_DIM = 128
EVENT_EMBED_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.2
EPOCHS = 10
MAX_LEN = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_MODELS = 3
MODEL_SEEDS = [42, 365, 2025]
TRAIN_PATH = '../data/train_augmented_data_1.csv'
VAL_PATH = '../data/val_augmented_data_1.csv'

def clean_text(text):
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'#(\w+)', r'HASHTAG_\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = html.unescape(text)
    text = re.sub(r'http\S+', 'URL', text)
    return text.lower().strip()

def tokenize(text):
    return WordPunctTokenizer().tokenize(clean_text(text))

def build_vocab(texts, min_freq=1):
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

def evaluate_single(transformer_model, loader):
    transformer_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_text, x_event, y in loader:
            x_text, x_event, y = x_text.to(DEVICE), x_event.to(DEVICE), y.to(DEVICE)
            logits = transformer_model(x_text, x_event)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def evaluate_ensemble(models, loader):
    for transformer_model in models:
        transformer_model.eval()
    all_probs, labels = [], []
    with torch.no_grad():
        for x_text, x_event, y in loader:
            x_text, x_event = x_text.to(DEVICE), x_event.to(DEVICE)
            batch_probs = []
            for transformer_model in models:
                logits = transformer_model(x_text, x_event)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs.append(avg_probs)
            labels.append(y.numpy())
    all_probs, labels = np.concatenate(all_probs), np.concatenate(labels)
    predictions = (all_probs > 0.5).astype(float)
    return np.mean(predictions == labels)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_model(transformer_model, optimizer, train_loader, val_loader, model_idx):
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    warmup_iters = min(500, len(train_loader))
    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, 0.1)
    best_val_acc, model_path = 0.0, f'best_transformer_rumor_detector_{model_idx}.pt'
    for epoch in range(EPOCHS):
        transformer_model.train()
        total_loss, total_samples = 0, 0
        for batch_idx, (x_text, x_event, y) in enumerate(train_loader):
            x_text, x_event, y = x_text.to(DEVICE), x_event.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = transformer_model(x_text, x_event)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
            optimizer.step()
            if epoch == 0 and batch_idx < warmup_iters:
                warmup_scheduler.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
        val_acc = evaluate_single(transformer_model, val_loader)
        avg_loss = total_loss / total_samples
        scheduler.step(val_acc)
        print(f'Model {model_idx} - Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f'Model {model_idx} - 保存新的最佳模型，验证准确率: {val_acc:.4f}')
    return model_path, best_val_acc

def main():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"训练集文件不存在: {TRAIN_PATH}")
    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(f"验证集文件不存在: {VAL_PATH}")
    train_df, val_df = pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH)
    required_columns = ['text', 'event', 'label']
    for col in required_columns:
        if col not in train_df.columns:
            print(f"警告: 训练集缺少列 '{col}'，尝试使用默认值")
            if col == 'event': train_df[col] = 0
            elif col == 'label': train_df[col] = 0
        if col not in val_df.columns:
            print(f"警告: 验证集缺少列 '{col}'，尝试使用默认值")
            if col == 'event': val_df[col] = 0
            elif col == 'label': val_df[col] = 0
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
    if 'label' in train_df.columns: print(f"训练集正负样本比例: {train_df['label'].mean():.4f}")
    if 'label' in val_df.columns: print(f"验证集正负样本比例: {val_df['label'].mean():.4f}")
    vocab = build_vocab(train_df['text'])
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print(f"词汇表大小: {len(vocab)}")
    num_events = max(train_df['event'].max(), val_df['event'].max()) + 1 if 'event' in train_df.columns and 'event' in val_df.columns else 1
    print(f"事件数量: {num_events}")
    train_set, val_set = RumorDataset(train_df, vocab), RumorDataset(val_df, vocab)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    model_paths, best_val_accs, trained_models = [], [], []
    print(f"\n开始训练 {NUM_MODELS} 个模型进行集成学习...")
    for i in range(NUM_MODELS):
        print(f"\n===== 训练模型 {i+1}/{NUM_MODELS} =====")
        seed = MODEL_SEEDS[i] if i < len(MODEL_SEEDS) else MODEL_SEEDS[0] + i
        torch.manual_seed(seed), np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        transformer_model = model.ImprovedTransformerRumorDetector(len(vocab), EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, num_events).to(DEVICE)
        total_params = sum(p.numel() for p in transformer_model.parameters())
        trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
        print(f"模型 {i+1} 总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        optimizer = optim.AdamW(transformer_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model_path, best_val_acc = train_model(transformer_model, optimizer, train_loader, val_loader, i+1)
        model_paths.append(model_path), best_val_accs.append(best_val_acc)
        print(f"模型 {i+1} 训练完成，最佳验证准确率: {best_val_acc:.4f}")
    print("\n各模型独立性能:")
    for i, acc in enumerate(best_val_accs): print(f"模型 {i+1}: 验证准确率 = {acc:.4f}")
    print(f"平均验证准确率: {np.mean(best_val_accs):.4f}")
    ensemble_models = []
    for path in model_paths:
        if os.path.exists(path):
            try:
                transformer_model = model.ImprovedTransformerRumorDetector(len(vocab), EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, num_events).to(DEVICE)
                checkpoint = torch.load(path)
                transformer_model.load_state_dict(checkpoint['model_state_dict'])
                ensemble_models.append(transformer_model)
                print(f"成功加载模型: {path}")
            except Exception as e: print(f"加载模型 {path} 失败: {e}")
        else: print(f"模型文件不存在: {path}")
    if len(ensemble_models) > 0:
        ensemble_acc = evaluate_ensemble(ensemble_models, val_loader)
        print(f"\n集成模型验证准确率: {ensemble_acc:.4f}")
        best_single, improvement = max(best_val_accs), ensemble_acc - max(best_val_accs)
        print(f"最佳单一模型准确率: {best_single:.4f}\n集成模型提升: {improvement:.4f} ({improvement*100:.2f}%)")
    else: print("没有可用的模型进行集成评估")

if __name__ == '__main__':
    main()