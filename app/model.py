import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import preprocess_text, encode_texts, compute_cosine_similarity


def compute_query_embeddings(user_query: str, encoder: SentenceTransformer) -> np.ndarray:
    """
    Convert a single user query into its embedding vector.
    
    Args:
        user_query (str): The original query text from the user.
        encoder (SentenceTransformer): A pre-trained sentence transformer model.
    
    Returns:
        np.ndarray: A 1D array of shape (embedding_dim,) representing the query.
    """
    clean_text = preprocess_text(user_query)
    embedding_array = encode_texts([clean_text], encoder)  # shape = (1, embedding_dim)
    query_embedding = embedding_array[0]  # 取出第一行 => shape (m,)
    return query_embedding


def recommend_indicators(user_query: str, indicator_ids: list, indicator_names: list, indicator_descriptions: list, indicator_embeddings: np.ndarray, indicator_dimensions: list, encoder: SentenceTransformer, top_n: int = 5) -> list:
    """
    1. Encode the user query into a vector of shape (embedding_dim,).
    2. Compute cosine similarity between this query vector and all indicator embeddings.
    3. Sort similarity scores in descending order and select the top_n indices.
    4. Gather and return the corresponding indicator information for those top indices.
    
    Args:
        user_query (str): The text query provided by the user.
        indicator_ids (list): List of indicator IDs.
        indicator_names (list): List of indicator names.
        indicator_descriptions (list): List of indicator descriptions.
        indicator_embeddings (np.ndarray): 2D array of shape (num_indicators, embedding_dim).
        indicator_dimensions (list): List of dimensions/categories for each indicator.
        encoder (SentenceTransformer): A pre-trained sentence transformer model.
        top_n (int, optional): Number of top matches to return. Defaults to 5.
    
    Returns:
        list: A list of dictionaries, each containing:
            - 'id': The indicator ID
            - 'name': The indicator name
            - 'description': The indicator description
            - 'dimension': The indicator’s dimension or category
            - 'similarity': The cosine similarity score as a float
    """
    # 1. Convert the user query into a vector (shape: embedding_dim,)
    query_emb = compute_query_embeddings(user_query, encoder)

    # 2. Compute cosine similarity between the query vector and all indicator vectors (result is shape: num_indicators)
    similarity_scores = compute_cosine_similarity(query_emb, indicator_embeddings)

    # 3. Get the indices of the top_n highest similarity scores (sorted in descending order)
    top_indices = np.argsort(-similarity_scores)[:top_n]

    # 4. Build and return a list of recommendation dicts
    recommendations = []
    for idx in top_indices:
        recommendations.append({'id': indicator_ids[idx], 'name': indicator_names[idx], 'description': indicator_descriptions[idx], 'dimensions': indicator_dimensions[idx], 'similarity': float(similarity_scores[idx])})

    return recommendations
