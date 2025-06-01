import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import random
import torch

MAX_LEN = 512

# 读取数据
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/val.csv')

# 加载 T5 模型和分词器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 't5-small'
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = t5_model.to(device)

# 初始化增强器
synonym_aug = naw.SynonymAug(aug_src='wordnet')
random_insert_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", aug_max=1)
random_delete_aug = naw.RandomWordAug(action="delete", aug_min=1, aug_max=1)

# 数据增强函数
def augment_data(text, methods=['random_insert']):
    augmented_texts = [text]  # 保留原始文本

    # 同义词替换
    if 'synonym' in methods:
        augmented_text = synonym_aug.augment(text)
        augmented_texts.append(augmented_text)

    # 随机插入
    if 'random_insert' in methods and random.random() < 0.5:
        augmented_text = random_insert_aug.augment(text)
        augmented_texts.append(augmented_text)

    # 随机删除
    if 'random_delete' in methods:
        augmented_text = random_delete_aug.augment(text)
        augmented_texts.append(augmented_text)

    # 使用 T5 模型改写文本
    if 't5' in methods and random.random() < 0.5: 
        try:
            inputs = t5_tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
            inputs = inputs.to(device)
            outputs = t5_model.generate(inputs, max_length=MAX_LEN, num_return_sequences=1)
            augmented_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            augmented_texts.append(augmented_text)
        except Exception as e:
            print(f"Error during T5 augmentation: {e}")
            augmented_texts.append(text)  # 如果出错，保留原始文本

    return augmented_texts

# 应用数据增强
all_augmented_data = []
for _, row in val_df.iterrows():
    id_val = row['id']
    text = row['text']
    label = row['label']
    event = row['event']

    if not isinstance(text, str) or pd.isnull(text):
        text = ""

    # 获取增强后的文本列表
    augmented_texts = augment_data(text)

    # 为每个增强后的文本创建新的数据行
    for aug_text in augmented_texts:
        all_augmented_data.append({
            'id': id_val,
            'text': aug_text,
            'label': label,
            'event': event
        })
        

augmented_df = pd.DataFrame(all_augmented_data)

# 保存增强后的数据
augmented_df.to_csv('./data/val_augmented_data_2.csv', index=False)