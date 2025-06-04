import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel

# 模型参数和训练配置
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 10
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理和加载
class RumorDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.events = df['event'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        event = self.events[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float),
            'event': torch.tensor(event, dtype=torch.long)
        }

# 定义 BERT 模型
class BertRumorDetector(nn.Module):
    def __init__(self, num_events):
        super(BertRumorDetector, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.event_embedding = nn.Embedding(num_events, 32)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, event):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]标记的输出
        event_features = self.event_embedding(event)
        combined = torch.cat((cls_output, event_features), dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(1)

# 训练函数
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        event = batch['event'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, event)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# 验证函数
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            event = batch['event'].to(DEVICE)

            outputs = model(input_ids, attention_mask, event)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# 主函数
def main():
    # 加载数据
    train_df = pd.read_csv('../data/train_augmented_data_1.csv')
    val_df = pd.read_csv('../data/val_augmented_data_1.csv')

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 创建数据集和数据加载器
    train_dataset = RumorDataset(train_df, tokenizer)
    val_dataset = RumorDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 获取事件数量
    num_events = max(train_df['event'].max(), val_df['event'].max()) + 1

    # 初始化模型、损失函数和优化器
    model = BertRumorDetector(num_events).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练和验证
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'bert_rumor_detector.pth')
    print('模型已保存')

if __name__ == '__main__':
    main()