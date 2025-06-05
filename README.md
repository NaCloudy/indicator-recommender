# Indicator Recommender

## Project Overview

**Indicator Recommender** is a lightweight system designed to automatically suggest relevant business indicators based on user input. It aims to support business intelligence (BI) data management by improving indicator discoverability and reusability.

## Features

- Load and store indicator metadata into a local SQLite database (using e-commerce metrics as examples)
- Compute and store text embeddings using a pretrained model to avoid redundant computation
- Recommend semantically similar indicators based on user input, along with similarity scores
- Support basic interactive command-line interface (CLI) for quick testing

## Requirements

- Python 3.9.7
- `pandas==1.3.4`
- `numpy==1.20.3`
- `sentence-transformers==4.1.0`

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/NaCloudy/indicator-recommender.git
cd indicator-recommender
```

2. **Create and activate a Python virtual environment:**

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the app:**

```bash
python app/main.py
```

## Example Usage

```text
Loading BERT model...
Loading database...
DB already exists, checking embeddings...
Loading embeddings from database...
=== E-commerce Indicator Recommender ===
Type 'exit' to quit at any time.

Enter your indicator demand (e.g., 'daily active users'):
```

> Input: `number of users active every day`

```text
Recommended Indicators:
1. Daily Active Users
   Description: Number of users who logged in and used the platform on a given day
   Dimensions: date,region
   Similarity Score: 0.815

2. Monthly Conversion Rate
   Description: Ratio of users who placed an order to those who visited the site in a month
   Dimensions: date,product
   Similarity Score: 0.364

3. Customer Retention Rate
   Description: Percentage of users who made repeat purchases over a period
   Dimensions: date,customer_segment
   Similarity Score: 0.359

4. Bounce Rate
   Description: Percentage of users who left the site after viewing only one page
   Dimensions: date,traffic_source
   Similarity Score: 0.298

5. Average Order Value
   Description: Average value of orders placed by users in a given period
   Dimensions: date,product
   Similarity Score: 0.259

Enter your indicator demand (e.g., 'daily active users'):
```

> Input: `exit`

```text
Exiting the recommender. Goodbye!
```

## Tech Stack

- **Text Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Similarity Metric:** Dot product (equivalent to cosine similarity after normalization)
- **Database:** SQLite (stores indicators and their embeddings)

## TODO

- ✅ Store indicator data in a SQLite database
- ✅ Precompute and store embeddings to avoid recomputation
- ⬜ Track user selections to enable offline model retraining (e.g., fine-tune BERT)
- ⬜ Expand the indicator dataset (currently only 10 examples)
- ⬜ Integrate with a large language model (e.g., Gemini) to refine top-3 results
- ⬜ Deploy as a web application for broader accessibility
