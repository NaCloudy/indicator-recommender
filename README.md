# Indicator Recommender

## Project Overview

**Indicator Recommender** is a lightweight system designed to automatically suggest relevant business indicators based on user input. It aims to support business intelligence (BI) data management by improving indicator discoverability and reusability.

## Features

- Load and store indicator metadata into a local SQLite database (using e-commerce metrics as examples)
- Compute and store text embeddings using a pretrained model to avoid redundant computation
- Recommend semantically similar indicators based on user input, along with similarity scores
- Collect and store user feedback on recommendations for future model improvement
- Support basic interactive command-line interface (CLI) for quick testing

## Tech Stack

- **Text Embedding Model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Similarity Metric:** Dot product (equivalent to cosine similarity after normalization)
- **Database:** SQLite (stores indicators, embeddings, and user feedback)

## Data Collection and Model Improvement

The system collects user feedback through a 5-point rating system:

- Users rate each recommended indicator based on relevance
- Feedback is stored in the database with timestamps
- Collected data can be used for:
  - Fine-tuning the embedding model
  - Analyzing recommendation quality
  - Understanding user needs and preferences
  - Improving future recommendations

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
Loading embedding model...
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

Please rate how relevant each recommendation is (1-5, where 5 is most relevant):

Rate 'Daily Active Users' (1-5):
```

> Input: `5`

```
Rate 'Monthly Conversion Rate' (1-5):
```

> Input: `1`

(and the rest 3 are rated...)

```
Thank you for your feedback! It will help improve future recommendations.
```

## TODO

- ✅ Store indicator data in a SQLite database
- ✅ Precompute and store embeddings to avoid recomputation
- ✅ Track user feedback for recommendations
- ⬜ Use collected feedback to fine-tune the embedding model
- ⬜ Expand the indicator dataset (currently only 10 examples)
- ⬜ Integrate with a large language model (e.g., Gemini) to refine top-3 results
- ⬜ Deploy as a web application for broader accessibility

## Development Progress

```mermaid
timeline
    title Project Progress
    Idea: 25-06-03 : Get the idea and plan
    Version 0.0: version 0.0.0 25-06-04: Minimum viable product, using csv and re-calculate in each query
    : version 0.0.1 25-06-04: Encapsulation of embedding and similarity calculation
    version 0.1 : version 0.1.0 25-06-05: Using database to store data
    : version 0.1.1 25-06-05: Adding "feedback" and "hit record" feature
```
