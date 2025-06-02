import torch
import pandas as pd
import model
import nltk
from nltk.tokenize import WordPunctTokenizer
import html
import re
import os
import json

nltk.download('punkt')

EMBEDDING_DIM = 128
EVENT_EMBED_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.2
MAX_LEN = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_text(text):
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'#(\w+)', r'HASHTAG_\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = html.unescape(text)
    text = re.sub(r'http\S+', 'URL', text)
    return text.lower().strip()

def tokenize(text):
    return WordPunctTokenizer().tokenize(clean_text(text))


class RumourDetectClass:
    def __init__(self, model_paths=['best_transformer_rumor_detector_1.pt','best_transformer_rumor_detector_2.pt','best_transformer_rumor_detector_3.pt'], data_path='../data/train.csv', vocab_path='vocab.json'):
        self.model_paths = model_paths
        self.train_df = pd.read_csv(data_path)
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        num_events = self.train_df['event'].max() + 1 if 'event' in self.train_df.columns else 1
        self.models = []
        for path in model_paths:
            if os.path.exists(path):
                try:
                    transformer_model = model.ImprovedTransformerRumorDetector(
                        vocab_size=len(self.vocab),
                        embedding_dim=EMBEDDING_DIM,
                        num_heads=NUM_HEADS,
                        num_layers=NUM_LAYERS,
                        ff_dim=FF_DIM,
                        num_events=num_events
                    ).to(DEVICE)
                    checkpoint = torch.load(path, map_location=DEVICE)
                    transformer_model.load_state_dict(checkpoint['model_state_dict'])
                    transformer_model.eval()
                    self.models.append(transformer_model)
                    print(f"成功加载模型: {path}")
                except Exception as e:
                    print(f"加载模型 {path} 失败: {e}")
            else:
                print(f"模型文件不存在: {path}")

    def preprocess(self, text):
        text = clean_text(text)
        tokens = tokenize(text)
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        if len(ids) < MAX_LEN:
            ids += [self.vocab['<PAD>']] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    def classify(self, text: str) -> int:
        text_ids = self.preprocess(text)
        event_id = torch.tensor([0], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits_list = [transformer_model(text_ids, event_id) for transformer_model in self.models]
            avg_logits = torch.mean(torch.stack(logits_list), dim=0)
            prob = torch.sigmoid(avg_logits)
        return 1 if prob.item() > 0.5 else 0

    def get_train_data(self):
        return self.train_df
    
    def test_csv(self, csv_path):
        # 读取CSV文件
        test_df = pd.read_csv(csv_path)

        test_correct = 0
        for i, row in test_df.iterrows():
            result = self.classify(row['text'])
            if (result == row['label']):
                test_correct += 1

        print(f"测试集准确率: {test_correct / len(test_df):.2%}")

# 使用示例
if __name__ == "__main__":
    detector = RumourDetectClass()
    
    # 测试单条文本
    result = detector.classify("Breaking: Earthquake hits California, magnitude 7.5!")
    print(result)
    
    # 测试CSV文件
    detector.test_csv(csv_path='../data/ai_generate.csv')