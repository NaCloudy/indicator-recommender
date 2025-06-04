# model.py

# 1. Load indicator data
# 2. Preprocess text fields (e.g., lowercasing, removing punctuation)
# 3. Use a pre-trained SentenceTransformer to encode both query and indicators
# 4. Compute cosine similarity between query vector and all indicator vectors
# 5. Return top-N matched indicators with similarity scores

# model.py

import pandas as pd
import numpy as np
#import re
from sentence_transformers import SentenceTransformer
from utils import preprocess_text

# def encode_texts(texts: list) -> np.ndarray:
#     """
#     临时版本：把每个词的 unicode 编码值取平均，返回一个固定长度的“向量”
#     （仅用于离线测试流程，不用于生产效果）
#     """
#     vecs = []
#     for text in texts:
#         codes = [ord(ch) for ch in text]
#         if len(codes) == 0:
#             vecs.append(np.zeros(1))
#         else:
#             vecs.append(np.array([sum(codes) / len(codes)]))
#     return np.vstack(vecs)

# def encode_texts(texts: list, model: SentenceTransformer) -> np.ndarray:
#     """
#     Use SentenceTransformer to convert a list of texts into embeddings.
#     - normalize_embeddings=True for cosine similarity
#     """
#     embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
#     return embeddings


def recommend_indicators(user_query: str, indicators_df: pd.DataFrame, model: SentenceTransformer, top_n: int = 5) -> list:
    """
    Recommend top-N indicators based on semantic similarity.

    Steps:
    1. Preprocess the user query.
    2. Encode query into an embedding.
    3. For each indicator, combine its name and definition, preprocess, and encode.
    4. Compute cosine similarity between the query embedding and all indicator embeddings.
    5. Return a list of dicts with keys: 'indicator_name', 'definition', 'dimensions', 'similarity'.
    """
    # 1. Preprocess and encode the user query
    query_processed = preprocess_text(user_query)
    query_embedding = model.encode([query_processed], convert_to_numpy=True, normalize_embeddings=True)[0]

    # 2. Prepare and encode indicator texts
    indicator_texts = []
    for _, row in indicators_df.iterrows():
        combined = f"{row['indicator_name']} {row['definition']}"
        processed = preprocess_text(combined)
        indicator_texts.append(processed)

    indicator_embeddings = model.encode(indicator_texts, convert_to_numpy=True, normalize_embeddings=True)

    # 3. Compute cosine similarity (dot product since embeddings are normalized)
    similarities = np.dot(indicator_embeddings, query_embedding)

    # 4. Get top_n indices sorted by similarity (descending)
    top_indices = np.argsort(-similarities)[:top_n]

    # 5. Collect the top-N recommendations
    recommendations = []
    for idx in top_indices:
        row = indicators_df.iloc[idx]
        recommendations.append({'indicator_name': row['indicator_name'], 'definition': row['definition'], 'dimensions': row['dimensions'], 'similarity': float(similarities[idx])})

    return recommendations
