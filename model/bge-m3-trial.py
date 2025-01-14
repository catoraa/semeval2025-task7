import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import json

import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取任务文件
posts = pd.read_csv("../preprocess_data/trial/posts.csv").fillna('')
fact_checks = pd.read_csv("../preprocess_data/trial/fact_checks.csv").fillna('')
predictions = pd.read_json("../preprocess_data/trial/trial_prediction.json", orient='index')
print('文件读取完成！\n')

# 根据posts_id查询对应的具体post句子
posts_sentences = posts.set_index('post_id')['post_text'].reindex(predictions.index).tolist()

# 获取所有的fact check句子及其ID
fact_check_sentences = fact_checks['fact_check_text'].tolist()
fact_check_ids = fact_checks['fact_check_id'].tolist()

# 加载模型
model = SentenceTransformer("BAAI/bge-m3").to(device)
model.max_seq_length = 512
print("模型加载完成！\n")

# 将模型设置为评估模式
model.eval()

# 计算posts中每个句子的嵌入
embeddings_posts = model.encode(posts_sentences, convert_to_tensor=True).cpu().numpy()
embeddings_fact_checks = model.encode(fact_check_sentences, convert_to_tensor=True).cpu().numpy()
print("嵌入计算完成！\n")

# 计算余弦相似度
similarities = cosine_similarity(embeddings_posts, embeddings_fact_checks)
print("相似度计算完成！\n")

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
print("答案已保存！\n")