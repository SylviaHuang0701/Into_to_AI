import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import random
import torch

MAX_LEN = 512

# 读取数据
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/val.csv')

import random
from collections import defaultdict

# 构建一个简单的同义词词典
synonyms = {
    'good': ['great', 'fine', 'excellent'],
    'bad': ['terrible', 'poor', 'awful'],
    'happy': ['joyful', 'content', 'pleased'],
    # 添加更多同义词
}

def synonym_replacement(text, n=1):
    words = text.split()
    augmented_words = words.copy()
    for _ in range(n):
        if len(words) > 0:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if word in synonyms:
                augmented_words[idx] = random.choice(synonyms[word])
    return ' '.join(augmented_words)

def random_insertion(text, n=1):
    words = text.split()
    for _ in range(n):
        if len(words) > 0:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            # 简单起见，这里直接插入原词，实际应用中可以插入相关词
            words.insert(idx, word)
    return ' '.join(words)

def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 0:
        return text
    augmented_words = []
    for word in words:
        if random.random() > p:
            augmented_words.append(word)
    if len(augmented_words) == 0:
        return random.choice(words)
    return ' '.join(augmented_words)

def random_swap(text, n=1):
    words = text.split()
    for _ in range(n):
        if len(words) >= 2:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def augment_data(text):
    augmented_texts = [text]
    
    # 同义词替换
    augmented_texts.append(synonym_replacement(text))
    
    # 随机插入
    augmented_texts.append(random_insertion(text))
    
    # 随机删除
    augmented_texts.append(random_deletion(text))
    
    # 随机交换
    augmented_texts.append(random_swap(text))
    
    return augmented_texts

# 应用数据增强
all_augmented_data = []
for _, row in val_df.iterrows():
    id_val = row['id']
    text = row['text']
    label = row['label']
    event = row['event']

    augmented_texts = augment_data(text)

    for aug_text in augmented_texts:
        all_augmented_data.append({
            'id': id_val,
            'text': aug_text,
            'label': label,
            'event': event
        })

augmented_df = pd.DataFrame(all_augmented_data)
augmented_df.to_csv('./data/val_augmented_data.csv', index=False)