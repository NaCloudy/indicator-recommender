# utils.py

# Utility functions used across the project:
# - Text cleaning
# - Similarity computation
# - Display formatting (optional)

### def clean_text(text: str) -> str:
###     """Basic text cleaning: lowercase, strip, remove symbols."""
###     pass
##
### def compute_cosine_similarity(query_vec, doc_vecs):
###     """Compute cosine similarity between a query vector and list ##of document vectors."""
###     pass
##
### def format_recommendations(recommendations: list):
###     """Format the output nicely for printing or logging."""
###     pass

# utils.py

import re
import numpy as np
import pandas as pd


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


def compute_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and an array of document vectors.
    
    Args:
        query_vec: 1D numpy array of shape (d,)
        doc_vecs: 2D numpy array of shape (n, d)
    
    Returns:
        1D numpy array of shape (n,) with cosine similarity scores.
    """
    # Ensure shapes
    if query_vec.ndim != 1 or doc_vecs.ndim != 2:
        raise ValueError("query_vec must be 1D and doc_vecs must be 2D numpy arrays")

    # Compute dot products
    dot_products = doc_vecs.dot(query_vec)

    # Compute norms
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    # Avoid division by zero
    denom = doc_norms * query_norm
    denom[denom == 0] = 1e-10

    return dot_products / denom


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
