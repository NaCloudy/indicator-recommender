# Utility functions used across the project:
# - Loading indicator data
# - Preprocessing text input
# - Encoding and similarity computation
# - Formatting recommendations
# - Managing SQLite database for indicators and embeddings

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import sqlite3
import pickle


def load_indicator_data(csv_path: str) -> pd.DataFrame:
    """
    Load indicator information from a CSV file into a DataFrame.
    Expected columns: ['id', 'indicator_name', 'definition', 'dimensions']
    
    Args:
        csv_path (str): Path to the CSV file containing indicator data.
    
    Returns:
        pd.DataFrame: DataFrame with indicator records.
    """
    df = pd.read_csv(csv_path)
    return df


def preprocess_text(text: str) -> str:
    """
    Clean and normalize input text:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Remove non-alphanumeric characters
    
    Args:
        text (str): Raw text string.
    
    Returns:
        str: Cleaned and normalized text.
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def encode_texts(texts: list, encoder: SentenceTransformer) -> np.ndarray:
    """
    Encode a list of text strings into embedding vectors using a SentenceTransformer model.
    The returned embeddings are normalized (unit vectors), so dot product equals cosine similarity.
    
    Args:
        texts (list): List of text strings to encode.
        encoder (SentenceTransformer): Pre-trained model for encoding.
    
    Returns:
        np.ndarray: Array of shape (len(texts), embedding_dimension).
    """
    embeddings = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def compute_cosine_similarity(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and multiple document vectors
    by taking their dot products (embeddings are assumed to be normalized).
    
    Args:
        query_vector (np.ndarray): 1D array of shape (embedding_dimension,).
        document_vectors (np.ndarray): 2D array of shape (num_documents, embedding_dimension).
    
    Returns:
        np.ndarray: 1D array of similarity scores with shape (num_documents,).
    """
    similarities = np.dot(document_vectors, query_vector)
    return similarities


def format_recommendations(recommendations: list) -> str:
    """
    Format a list of recommendation dictionaries into a readable multi-line string.
    
    Each dictionary in recommendations should have keys:
      - 'name'
      - 'description'
      - 'dimension'
      - 'similarity'
    
    Args:
        recommendations (list): List of dicts, each containing recommendation details.
    
    Returns:
        str: A formatted string with one recommendation per block.
    """
    lines = []
    for idx, rec in enumerate(recommendations, start=1):
        name = rec.get('name', 'N/A')
        description = rec.get('description', 'N/A')
        dimensions = rec.get('dimensions', 'N/A')
        similarity = rec.get('similarity', 0.0)
        lines.append(f"{idx}. {name}\n"
                     f"   Description: {description}\n"
                     f"   Dimensions: {dimensions}\n"
                     f"   Similarity Score: {similarity:.3f}")
    return "\n\n".join(lines)


def create_connection(db_path: str) -> sqlite3.Connection:
    """
    Create a connection to a SQLite database (or open it if it already exists).
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Returns:
        sqlite3.Connection: SQLite connection object.
    """
    connection = sqlite3.connect(db_path)
    return connection


def create_tables(db_path: str):
    """
    Create two tables in the SQLite database if they do not already exist:
      1. ind (id, name, description, dimension)
      2. emb (indicator_id, embedding_blob)
    
    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = create_connection(db_path)
    c = conn.cursor()
    # Table for indicator metadata
    c.execute('''
        CREATE TABLE IF NOT EXISTS ind (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            dimensions TEXT NOT NULL
        )
    ''')
    # Table for serialized embeddings
    c.execute('''
        CREATE TABLE IF NOT EXISTS emb (
            indicator_id INTEGER NOT NULL,
            embedding_blob BLOB NOT NULL,
            FOREIGN KEY(indicator_id) REFERENCES ind(id)
        )
    ''')
    conn.commit()
    conn.close()


def insert_indicators_from_csv(csv_path: str, db_path: str):
    """
    Read indicator records from a CSV file and insert them into the 'ind' table.
    The CSV is expected to have columns: ['name', 'description', 'dimension'].
    Existing records in 'ind' will be removed before insertion.
    
    Args:
        csv_path (str): Path to the CSV file containing indicator data.
        db_path (str): Path to the SQLite database file.
    """
    dataframe = pd.read_csv(csv_path)
    conn = create_connection(db_path)
    cursor = conn.cursor()

    # Clear existing data if reinitializing
    cursor.execute("DELETE FROM ind")
    conn.commit()

    # Insert each row into the ind table
    for _, row in dataframe.iterrows():
        name = row['name']
        description = row['description']
        dimensions = row['dimensions']
        cursor.execute("INSERT INTO ind (name, description, dimensions) VALUES (?, ?, ?)", (name, description, dimensions))
    conn.commit()
    conn.close()


def compute_and_store_embeddings(db_path: str, encoder: SentenceTransformer):
    """
    1. Read all (id, name, description) entries from the 'ind' table.
    2. Concatenate name and description, preprocess, and encode each into an embedding.
    3. Serialize each embedding with pickle and store as a blob in the 'emb' table.
    
    Args:
        db_path (str): Path to the SQLite database file.
        encoder (SentenceTransformer): Pre-trained model for encoding text.
    
    Raises:
        RuntimeError: If the 'ind' table is empty.
    """
    # 1. read ind table
    conn = create_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, description FROM ind")
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        raise RuntimeError("The 'ind' table is empty. Insert indicator data first.")

    indicator_ids = [r[0] for r in rows]
    texts = [f"{r[1]} {r[2]}" for r in rows]  # Combine name and description

    # Preprocess texts and encode them
    cleaned_texts = [preprocess_text(text) for text in texts]
    embeddings = encode_texts(cleaned_texts, encoder)

    # Insert serialized embeddings into the emb table
    for ind_id, vector in zip(indicator_ids, embeddings):
        blob = pickle.dumps(vector)
        cursor.execute("INSERT INTO emb (indicator_id, embedding_blob) VALUES (?, ?)", (ind_id, blob))

    conn.commit()
    conn.close()


def load_embeddings_from_db(db_path: str) -> tuple:
    """
    Load all indicator metadata and their embeddings from the database.
    
    Returns:
        tuple: (ids, names, descriptions, embeddings_array, dimensions)
            - ids (list of int)
            - names (list of str)
            - descriptions (list of str)
            - embeddings_array (np.ndarray of shape (num_indicators, embedding_dim))
            - dimensions (list of str)
    """
    conn = create_connection(db_path)
    cursor = conn.cursor()

    # Retrieve indicator metadata
    cursor.execute("SELECT id, name, description, dimensions FROM ind")
    ind_rows = cursor.fetchall()

    ids = [r[0] for r in ind_rows]
    names = [r[1] for r in ind_rows]
    descriptions = [r[2] for r in ind_rows]
    dimensions = [r[3] for r in ind_rows]

    # Retrieve serialized embeddings
    cursor.execute("SELECT embedding_blob FROM emb")
    emb_rows = cursor.fetchall()

    embeddings = []
    for row in emb_rows:
        blob = row[0]
        vector = pickle.loads(blob)
        embeddings.append(vector)

    conn.close()
    embeddings_array = np.array(embeddings)
    return ids, names, descriptions, embeddings_array, dimensions
