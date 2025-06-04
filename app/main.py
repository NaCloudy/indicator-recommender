# 1. Load data and model
# 2. Take user query input
# 3. Call recommend_indicators
# 4. Display results in a readable way

import pandas as pd
from sentence_transformers import SentenceTransformer
from model import recommend_indicators
from utils import load_indicator_data, format_recommendations


def main():
    print("=== E-commerce Indicator Recommender ===")
    print("Type 'exit' to quit at any time.")

    # 1. Load indicator data from CSV
    indicators_df = load_indicator_data('./data/indicators.csv')

    # 2. Load the pre-trained SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    while True:
        # 3. Prompt user for input
        user_query = input("\nEnter your indicator demand (e.g., 'daily active users'): ").strip()
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break

        if not user_query:
            print("Please enter a non-empty query.")
            continue

        # 4. Get Top-5 recommended indicators
        recommendations = recommend_indicators(user_query, indicators_df, model, top_n=5)

        # 5. Format and display the results
        print("\nRecommended Indicators:")
        print(format_recommendations(recommendations))


if __name__ == "__main__":
    main()
