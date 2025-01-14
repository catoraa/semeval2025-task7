import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import json
import wandb
import torch
from datasets import Dataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer

wandb.login(key="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取任务文件
posts = pd.read_csv("../preprocess_data/trial/posts.csv").fillna('')
fact_checks = pd.read_csv("../preprocess_data/trial/fact_checks.csv").fillna('')
pairs = pd.read_csv('../preprocess_data/trial/pairs.csv').fillna('')
predictions = pd.read_json("../preprocess_data/trial/trial_predictions.json", orient='index')
print('数据读取完成!')

# 根据posts_id查询对应的具体post句子
posts_sentences = posts.set_index('post_id')['post_text'].reindex(predictions.index).tolist()

# 获取所有的fact check句子及其ID
fact_check_sentences = fact_checks['fact_check_text'].tolist()
fact_check_ids = fact_checks['fact_check_id'].tolist()

# 格式化训练数据
train_dataset = Dataset.from_dict({
    "sentence1": pairs["post"].tolist(),
    "sentence2": pairs["fact_check"].tolist(),
})
print('数据处理完成!')

# 加载模型
model = SentenceTransformer("BAAI/bge-m3").to(device)
model.max_seq_length = 512
loss = MultipleNegativesRankingLoss(model)
print("模型加载完成!")

# 模型微调
args = SentenceTransformerTrainingArguments(
    output_dir="./checkpoint",        # output directory and hugging face model ID
    num_train_epochs=1,                         # number of epochs
    per_device_train_batch_size=12,             # train batch size
    per_device_eval_batch_size=12,              # evaluation batch size
    warmup_ratio=0.1,                           # warmup ratio
    learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    fp16=False,                                  # use fp16 precision
    bf16=False,                                  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="no",                         # evaluate after each epoch
    save_strategy="no",                         # save after each epoch
    logging_steps=100,                          # log every 10 steps
    run_name="bge-m3-trial-finetuned",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
print("模型微调开始!")
trainer.train()
print("模型微调完成!")

# 将模型设置为评估模式
model.eval()

# 计算posts中每个句子的嵌入
embeddings_posts = model.encode(posts_sentences, convert_to_tensor=True).cpu().numpy()
embeddings_fact_checks = model.encode(fact_check_sentences, convert_to_tensor=True).cpu().numpy()
print("嵌入计算完成!")

# 计算余弦相似度
similarities = cosine_similarity(embeddings_posts, embeddings_fact_checks)
print("相似度计算完成!")

# 设定Top
top_k = 5

# 计算并保存答案
result = {}
for i, post_id in enumerate(predictions.index):
    post_similarities = similarities[i]
    top_indices = np.argsort(post_similarities)[-top_k:]
    top_k_fact_checks = [fact_check_ids[idx] for idx in top_indices]
    result[post_id] = top_k_fact_checks

with open("../output/dev/trial_predictions.json", "w") as f:
    json.dump(result, f)

print("答案已保存!")