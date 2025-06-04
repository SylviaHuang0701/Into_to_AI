import torch
import pandas as pd
import model
import nltk
from nltk.tokenize import WordPunctTokenizer
import html
import re
import os
import json
import argparse
from train import clean_text, tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

EMBEDDING_DIM = 128
EVENT_EMBED_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.2
MAX_LEN = 64
NUM_EVENTS = 7
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RumourDetectClass:
    def __init__(self, model_paths=None, vocab_path='vocab.json', use_event=True):
        """初始化谣言检测器"""
        if model_paths is None:
            # 默认使用modified模型
            model_paths = [
                'transformer_rumor_detector_withevent_1.pt',
                'transformer_rumor_detector_withevent_2.pt',
                'transformer_rumor_detector_withevent_3.pt'
            ] if use_event else [
                'transformer_rumor_detector_1.pt',
                'transformer_rumor_detector_2.pt',
                'transformer_rumor_detector_3.pt'
            ]
            
        self.model_paths = model_paths
        self.use_event = use_event
        print(f"使用{'带' if use_event else '不带'}事件信息的模型")
        
        # 加载词表
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            print(f"成功加载词表，大小: {len(self.vocab)}")
        except Exception as e:
            raise RuntimeError(f"加载词表失败: {e}")

        # 加载模型
        self.models = []
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    transformer_model = model.TransformerRumorDetector(
                        vocab_size=len(self.vocab),
                        embedding_dim=EMBEDDING_DIM,
                        num_heads=NUM_HEADS,
                        num_layers=NUM_LAYERS,
                        ff_dim=FF_DIM,
                        num_events=NUM_EVENTS,
                        use_event=self.use_event
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
        
        if not self.models:
            raise RuntimeError("没有成功加载任何模型")

    def preprocess(self, text):
        """预处理文本"""
        text = clean_text(text)
        tokens = tokenize(text)
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        if len(ids) < MAX_LEN:
            ids += [self.vocab['<PAD>']] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    def classify(self, text: str, event_id=None) -> int:
        """
        对输入的文本进行谣言检测
        Args:
            text: 输入的文本字符串
            event_id: 事件ID（可选）
        Returns:
            int: 预测的类别（0表示非谣言，1表示谣言）
        """
        text_ids = self.preprocess(text)
        
        # 如果没有提供event_id但模型需要事件信息，使用默认值0
        if self.use_event and event_id is None:
            event_id = 0
            print("警告：模型需要事件信息但未提供，使用默认值0")
        
        with torch.no_grad():
            logits_list = []
            for transformer_model in self.models:
                if self.use_event:
                    event_tensor = torch.tensor([event_id], dtype=torch.long).to(DEVICE)
                    logits = transformer_model(text_ids, event_tensor)
                else:
                    logits = transformer_model(text_ids)
                logits_list.append(logits)
            
            # 确保有预测结果
            if not logits_list:
                raise RuntimeError("没有有效的预测结果")
            
            avg_logits = torch.mean(torch.stack(logits_list), dim=0)
            prob = torch.sigmoid(avg_logits)
            return 1 if prob.item() > 0.5 else 0

    def test_csv(self, csv_path):
        """测试CSV文件中的样本"""
        try:
            test_df = pd.read_csv(csv_path)
            total = len(test_df)
            correct = 0
            
            print(f"开始测试 {total} 个样本")
            
            for i, row in test_df.iterrows():
                if i % 100 == 0:
                    print(f"已处理: {i}/{total}")
                
                # 如果数据集中有事件信息且模型使用事件信息，则使用数据集中的事件ID
                event_id = row['event'] if 'event' in test_df.columns and self.use_event else None
                result = self.classify(row['text'], event_id)
                
                if result == row['label']:
                    correct += 1
            
            accuracy = correct / total
            print(f"测试完成，准确率: {accuracy:.2%}")
            return accuracy
            
        except Exception as e:
            print(f"测试过程出错: {e}")
            return 0.0

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='谣言检测模型测试程序')
    parser.add_argument('--use-event', type=int, choices=[0, 1], default=0,
                      help='是否使用事件信息 (0: 不使用, 1: 使用)')
    parser.add_argument('--test-file', type=str, default='../data/testing.csv',
                      help='测试数据文件路径')
    return parser.parse_args()

# 使用示例
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 初始化检测器
    detector = RumourDetectClass(
        use_event=bool(args.use_event)
    )
    
    # 测试数据集
    detector.test_csv(args.test_file)