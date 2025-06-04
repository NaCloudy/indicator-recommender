# Utility functions used across the project:
# - Loading indicator data
# - Preprocessing text input
# - Similarity computation
# - Display formatting (optional)

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_indicator_data(path: str) -> pd.DataFrame:
    """
    Load indicators from a CSV file into a DataFrame.
    Expected columns: ['id', 'indicator_name', 'definition', 'dimensions']
    """
    df = pd.read_csv(path)
    return df


def preprocess_text(text: str) -> str:
    """
    Clean and normalize input text:
    - Lowercase
    - Strip whitespace
    - Remove non-alphanumeric characters
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def encode_texts(texts: list, model: SentenceTransformer) -> np.ndarray:
    """
    把一个字符串列表 texts 编码成 embeddings：
      - 返回 shape = (len(texts), embedding_dim) 的 numpy 数组
      - normalize_embeddings=True => 生成的向量已经是单位向量，可以直接做点积来算 cosine 相似度
    """
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def compute_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    计算 query_vec（shape=(m,)） 与 doc_vecs（shape=(n, m)）之间的点积；
    因为 embeddings 已经 normalize 过，点积等价于 cosine similarity。
    返回值形状为 (n,)
    """
    # 注意 np.dot(doc_vecs, query_vec) 会得到一个 (n,) 的数组
    similarities = np.dot(doc_vecs, query_vec)
    return similarities


def format_recommendations(recommendations: list) -> str:
    """
    Format the list of recommendation dicts into a readable multi-line string.
    
    Each dict in recommendations should have keys:
      - 'indicator_name'
      - 'definition'
      - 'dimensions'
      - 'similarity'
    
    Returns:
        A single string with one recommendation per line.
    """
    lines = []
    for idx, rec in enumerate(recommendations, start=1):
        name = rec.get('indicator_name', 'N/A')
        definition = rec.get('definition', 'N/A')
        dimensions = rec.get('dimensions', 'N/A')
        similarity = rec.get('similarity', 0.0)
        lines.append(f"{idx}. {name}\n"
                     f"   Definition: {definition}\n"
                     f"   Dimensions: {dimensions}\n"
                     f"   Similarity Score: {similarity:.3f}")
    return "\n\n".join(lines)
