import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import re
import html

MAX_LEN = 512

train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/val.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

synonym_aug = naw.SynonymAug(aug_src='wordnet')
random_insert_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", aug_max=1)
random_delete_aug = naw.RandomWordAug(action="delete", aug_min=1, aug_max=1)

def clean_text(text):
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'#(\w+)', r'HASHTAG_\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = html.unescape(text)
    text = re.sub(r'http\S+', 'URL', text)
    return text.lower().strip()

def augment_data(text, methods=['synonym', 'random_insert', 'random_delete']):
    augmented_texts = [text]

    # BART同义词替换和句子重组
    if 'synonym' in methods:
        inputs = bart_tokenizer.encode( text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
        inputs = inputs.to(device)
        outputs = bart_model.generate(
            inputs,
            max_length=MAX_LEN,
            num_return_sequences=1,
            temperature=1.0,
            num_beams=5,
            min_length=10,
            early_stopping=True
        )
        augmented_text = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        augmented_texts.append(augmented_text)

    # 随机删除
    if 'random_delete' in methods:
        augmented_text = random_delete_aug.augment(text)
        augmented_texts.append(augmented_text)

    return augmented_texts

all_augmented_data = []
for _, row in val_df.iterrows():
    id_val = row['id']
    text = row['text']
    label = row['label']
    event = row['event']

    if not isinstance(text, str) or pd.isnull(text):
        text = ""

    cleaned_text = clean_text(text)
    augmented_texts = augment_data(cleaned_text)

    for aug_text in augmented_texts:
        all_augmented_data.append({
            'id': id_val,
            'text': aug_text,
            'label': label,
            'event': event
        })

augmented_df = pd.DataFrame(all_augmented_data)
augmented_df.to_csv('./data/val_augmented_data.csv', index=False)