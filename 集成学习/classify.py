import torch
import pandas as pd
import model
import nltk
import os
import json
import argparse
from train import clean_text, tokenize
from visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

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
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RumourDetectClass:
    def __init__(self, model_paths=None, vocab_path='vocab.json', use_event=True):
        if model_paths is None:
            # 默认使用加入新数据的训练集训出的模型
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
        
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            print(f"成功加载词表，大小: {len(self.vocab)}")
        except Exception as e:
            raise RuntimeError(f"加载词表失败: {e}")
        
        try:
            with open('num_events.json', 'r') as f:
                num_events_data = json.load(f)
                self.num_events = num_events_data.get('num_events', 1)
            print(f"成功加载 num_events: {self.num_events}")
        except Exception as e:
            print(f"加载 num_events 失败: {e}")
            self.num_events = 7

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
                        num_events=self.num_events,
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
        text = clean_text(text)
        tokens = tokenize(text)
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        if len(ids) < MAX_LEN:
            ids += [self.vocab['<PAD>']] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    def classify(self, text: str, event_id=None) -> tuple:
        """
        对输入的文本进行谣言检测
        Args:
            text: 输入的文本字符串
            event_id: 事件ID，默认模型不使用event信息训练，因此调用时不可加入event
        Returns:
            int: 预测的类别（0表示非谣言，1表示谣言）
        
        为绘制混淆矩阵和ROC曲线，原来classify返回的是tuple(类别，概率），但现在只返回类别
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
            
            y_true = []
            y_pred = []
            y_prob = []
            
            for i, row in test_df.iterrows():
                if i % 100 == 0:
                    print(f"已处理: {i}/{total}")
                event_id = row['event'] if 'event' in test_df.columns and self.use_event else None
                result = self.classify(row['text'], event_id)

                y_true.append(row['label'])
                y_pred.append(result)
                
                if result == row['label']:
                    correct += 1
            
            accuracy = correct / total
            print(f"测试完成，准确率: {accuracy:.2%}")
            
            plot_confusion_matrix(y_true, y_pred)
            
            return accuracy
            
        except Exception as e:
            print(f"测试过程出错: {e}")
            return 0.0

def parse_args():
    parser = argparse.ArgumentParser(description='谣言检测模型测试程序')
    parser.add_argument('--use-event',action='store_true',help='是否使用事件信息')
    parser.add_argument('--test-file', type=str, default='../data/testing.csv',help='测试数据文件路径')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    detector = RumourDetectClass(
        use_event=bool(args.use_event)
    )
    
    accuracy = detector.test_csv(args.test_file)
    
    text = "#rcmp to hold news conference on #ottawa shootings at 2 pm et, 11 am pt. watch live coverage @ URL"
    result = detector.classify(text)
    print(result)