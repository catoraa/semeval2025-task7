import ast
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
from transformers import Trainer, TrainingArguments

class FactCheckDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# 加载数据
posts = pd.read_csv('../sample_data/trial_posts.csv')
fact_checks = pd.read_csv('../sample_data/trial_fact_checks.csv')
mapping = pd.read_csv('../sample_data/trial_data_mapping.csv')

# 文件预处理 将\n转换为\nn避免歧义
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s
for col in ['claim', 'instances', 'title']:
    fact_checks[col] = fact_checks[col].apply(parse_col)
for col in ['instances', 'ocr', 'verdicts', 'text']:
    posts[col] = posts[col].apply(parse_col)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 初始化tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('xlm-roberta-base')


    # 为每个帖子和事实核查对创建编码和标签
    def create_dataset(is_train):
        # 从mapping文件中获取强相关的帖子和事实核查对
        if is_train:
            data = mapping
        else:
            # 从剩余的帖子和事实核查中获取未配对的对
            all_posts = posts[['post_id']].drop_duplicates()
            all_fact_checks = fact_checks[['fact_check_id']].drop_duplicates()
            data = pd.merge(all_posts, all_fact_checks, how='cross').drop_duplicates(subset=['post_id', 'fact_check_id'])

        # 创建正样本和负样本标签
        labels = np.zeros(len(data), dtype=int)

        if is_train:
            labels[data['post_id'].isin(mapping['post_id'])] = 1

        # 对帖子和事实核查进行编码
        encodings = tokenizer(
            data['text'].tolist(),
            data['claim'].tolist(),
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return FactCheckDataset(encodings, labels)


    # 构建训练和验证数据集
    train_dataset = create_dataset(is_train=True)
    val_dataset = create_dataset(is_train=False)

    # 构建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 构建DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = SentenceTransformer.from_pretrained('roberta-base', num_labels=2)

    training_args = TrainingArguments(
        output_dir='../result',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        learning_rate=5e-6,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.evaluate()
