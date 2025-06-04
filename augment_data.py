import nltk
from nltk.corpus import wordnet
import random
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def get_synonyms(word):
    """
    获取一个单词的同义词列表
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def synonym_replacement(text, n_replacements):
    """
    对文本进行同义词替换增强
    """
    words = word_tokenize(text)
    augmented_words = words.copy()
    unique_words = list(set(words))
    n_replaced = 0
    for word in unique_words:
        if n_replaced >= n_replacements:
            break
        synonyms = get_synonyms(word)
        if len(synonyms) > 0 and word != "":
            random_synonym = random.choice(synonyms)
            augmented_words = [random_synonym if w == word else w for w in augmented_words]
            n_replaced += 1
    augmented_text = ' '.join(augmented_words)
    return augmented_text

import random

def sentence_restructuring(text):
    """
    对文本进行句子重组增强
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return text
    # 随机交换两个相邻的句子
    i = random.randint(0, len(sentences)-2)
    sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
    augmented_text = ' '.join(sentences)
    return augmented_text

df = pd.read_csv('./data/train.csv')
# 应用数据增强
all_augmented_data = []
for _, row in df.iterrows():
    id_val = row['id']
    text = row['text']
    label = row['label']
    event = row['event']

    if not isinstance(text, str) or pd.isnull(text):
        text = ""

    # 先保留原始文本
    all_augmented_data.append({
        'id': id_val,
        'text': text,
        'label': label,
        'event': event
    })

    # 获取增强后的文本
    augmented_text = synonym_replacement(text, 2)

    # 创建新的数据行以保存增强后的文本
    all_augmented_data.append({
        'id': id_val,
        'text': augmented_text,
        'label': label,
        'event': event
    })

augmented_df = pd.DataFrame(all_augmented_data)

# 保存增强后的数据
augmented_df.to_csv('./data/train_augmented_data_simple.csv', index=False)