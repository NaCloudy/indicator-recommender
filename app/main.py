import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from model import recommend_indicators
from utils import format_recommendations, create_tables, insert_indicators_from_csv, compute_and_store_embeddings, load_embeddings_from_db, store_user_feedback

# Paths for the source CSV and the SQLite database
INDICATOR_CSV_PATH = "./data/indicators.csv"
DB_PATH = "./data/indicators.db"
MODEL_NAME = "all-MiniLM-L6-v2"


def initialize_database(encoder: SentenceTransformer):
    """
    Initialize the SQLite database for indicators and embeddings.

    1. If the database file does not exist:
       - Create the tables
       - Load indicator records from the CSV
       - Compute and store embeddings using the provided encoder_model

    2. If the database already exists:
       - Print a message to indicate that we will skip re-creation
    """
    if not os.path.exists(DB_PATH):
        create_tables(DB_PATH)
        insert_indicators_from_csv(INDICATOR_CSV_PATH, DB_PATH)
        compute_and_store_embeddings(DB_PATH, encoder)
    else:
        print("Database already exists. Skipping initialization.")


def get_user_feedback(recommendations: list) -> dict:
    """
    Collect user feedback for the recommended indicators.
    
    Args:
        recommendations (list): List of recommendation dictionaries.
    
    Returns:
        dict: Dictionary mapping indicator IDs to feedback scores.
    """
    feedback = {}
    print("\nPlease rate how relevant each recommendation is (1-5, where 5 is most relevant):")
    for rec in recommendations:
        while True:
            try:
                score = int(input(f"\nRate '{rec['name']}' (1-5): "))
                if 1 <= score <= 5:
                    feedback[rec['id']] = score
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
    return feedback


def main():
    # 1. Load pre-trained SentenceTransformer model
    print("Loading BERT model...")
    os.environ['http_proxy'] = "http://127.0.0.1:7890"
    os.environ['https_proxy'] = "http://127.0.0.1:7890"
    encoder_model = SentenceTransformer('sentence-transformers/' + MODEL_NAME)

    # 2. Initialize the database and ensure indicator data is loaded
    print("Initializing or loading database...")
    initialize_database(encoder_model)

    # 3. Load the embedding data from the database
    print("Loading embeddings from database...")
    indicator_ids, indicator_names, indicator_descriptions, indicator_embeddings, indicator_dimensions = load_embeddings_from_db(DB_PATH)

    print("=== E-commerce Indicator Recommender ===")
    print("Type 'exit' or 'quit' to end the program.")

    while True:
        # 4. Prompt the user for an indicator-related query
        user_query = input("\nEnter your indicator query (e.g., 'daily active users'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting the recommender. Goodbye!")
            break

        if not user_query:
            print("Please enter a non-empty query.")
            continue

        # 5. Get Top-5 recommended indicators
        recommendations = recommend_indicators(user_query, indicator_ids, indicator_names, indicator_descriptions, indicator_embeddings, indicator_dimensions, encoder_model, top_n=5)

        # 6. Format and display the results
        print("\nRecommended Indicators:")
        print(format_recommendations(recommendations))

        # 7. Collect and store user feedback
        feedback = get_user_feedback(recommendations)
        for indicator_id, score in feedback.items():
            store_user_feedback(DB_PATH, user_query, indicator_id, score)
        print("\nThank you for your feedback! It will help improve future recommendations.")


if __name__ == "__main__":
    main()
