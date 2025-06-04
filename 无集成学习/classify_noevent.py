import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
import model
import nltk
from nltk.tokenize import WordPunctTokenizer
import html
import re
import json
from train import clean_text, tokenize

nltk.download('punkt')

# 超参数设置
EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
MAX_LEN = 128
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RumourDetectClass:
    def __init__(self, model_path='transformer_rumor_detector2.pt', vocab_path='vocab.json'):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.model = model.TransformerRumorDetector(
            vocab_size=len(self.vocab),
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

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
        with torch.no_grad():
            logits = self.model(text_ids)
            prob = torch.sigmoid(logits)
        return 1 if prob.item() > 0.5 else 0
    
    def test_csv(self, csv_path):
        test_df = pd.read_csv(csv_path)
        test_correct = 0
        for _, row in test_df.iterrows():
            result = self.classify(row['text'])
            if result == row['label']:
                test_correct += 1
        print(f"测试集准确率: {test_correct / len(test_df):.2%}")

if __name__ == "__main__":
    detector = RumourDetectClass()
    result = detector.classify("Breaking: Earthquake hits California, magnitude 7.5!")
    print(result)
    detector.test_csv(csv_path='../data/testing.csv')