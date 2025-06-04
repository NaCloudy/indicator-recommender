# 1. Load indicator data
# 2. Preprocess text fields (e.g., lowercasing, removing punctuation)
# 3. Use a pre-trained SentenceTransformer to encode both query and indicators
# 4. Compute cosine similarity between query vector and all indicator vectors
# 5. Return top-N matched indicators with similarity scores

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import preprocess_text, encode_texts, compute_cosine_similarity


def compute_ind_embeddings(indicators_df: pd.DataFrame, model: SentenceTransformer):
    """
    第一个 embedding 操作：
    - 输入 indicators_df: 必须包含 'id', 'indicator_name', 'definition' 这几列
    - 输出两个对象：
        1. ids: 一个长度为 n 的一维数组（或列表），对应每条指标的 id
        2. embeddings: 一个 shape=(n, m) 的 numpy 数组，m 是 embedding 维度
    """
    # 1. 先把 n 条记录的文本都拼接好，并做清洗
    texts = []
    for _, row in indicators_df.iterrows():
        combined = f"{row['indicator_name']} {row['definition']}"
        texts.append(preprocess_text(combined))

    # 2. 批量调用 encode，得到 (n, m) 的 numpy 数组
    embeddings = encode_texts(texts, model)  # 形状 (n, m)

    # 3. 同时把原始的 id 列取出来，变成长度为 n 的列表或一维数组
    ids = indicators_df['id'].values  # shape = (n,)

    return ids, embeddings


def compute_nl_embeddings(user_query: str, model: SentenceTransformer) -> np.ndarray:
    """
    第二个 embedding 操作：针对单条用户 query
    - 输入 user_query: 原始文本
    - 输出 query_embedding: 一个 shape=(m,) 的向量
    """
    processed = preprocess_text(user_query)
    emb_array = encode_texts([processed], model)  # shape = (1, m)
    query_embedding = emb_array[0]  # 取出第一行 => shape (m,)
    return query_embedding


def recommend_indicators(user_query: str, indicators_df: pd.DataFrame, model: SentenceTransformer, top_n: int = 5) -> list:
    """
    1. 先把用户 query 编码成 (m,) 向量
    2. 再把所有指标的文本批量编码成 (n, m) 矩阵
    3. 计算 (n, m) 和 (m,) 的点积，得到 (n,) 相似度数组
    4. 排序后取 top_n 的索引
    5. 根据索引从原 indicators_df 里取对应的行，封装成推荐列表返回
    """
    # 1. 用户 query embedding (m,)
    query_emb = compute_nl_embeddings(user_query, model)

    # 2. 指标整体 embedding: ids (n,), ind_embs (n, m)
    ids, ind_embs = compute_ind_embeddings(indicators_df, model)

    # 3. 计算相似度 (n,)
    similarities = compute_cosine_similarity(query_emb, ind_embs)

    # 4. 取前 top_n 个 index
    top_indices = np.argsort(-similarities)[:top_n]  # 降序排序后前 top_n

    # 5. 封装返回结果
    recommendations = []
    for idx in top_indices:
        row = indicators_df.iloc[idx]
        recommendations.append({'indicator_name': row['indicator_name'], 'definition': row['definition'], 'dimensions': row['dimensions'], 'similarity': float(similarities[idx])})

    return recommendations
