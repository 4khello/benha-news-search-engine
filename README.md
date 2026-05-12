# Benha News Search Engine

A Flask-based news search engine that indexes and retrieves news articles using classic Information Retrieval techniques.

The project loads a news dataset, preprocesses article text, builds an inverted index, ranks results using TF-IDF, and provides a clean web interface for searching and filtering news by subject, title, and publication date.

## Features

- Web search interface built with Flask
- Text preprocessing using tokenization, stop-word removal, and lemmatization
- Inverted index construction
- TF-IDF ranking
- Query expansion using a small synonym map
- Pseudo relevance feedback from top-ranked documents
- Filters by subject, title, and date range
- Search result cards with article title, summary, image, date, subject, and score
- Evaluation page with sample queries and response-time measurements
- Collection statistics such as number of documents, terms, postings, and tokens

## Tech Stack

- Python
- Flask
- Pandas
- NLTK
- HTML
- CSS

## Project Structure

```text
benha-news-search-engine/
├── app.py
├── requirements.txt
├── data/
│   └── NEWS.csv
├── static/
│   ├── logo.png
│   └── style.css
└── templates/
    └── index.html
```

## Dataset

The dataset is stored in:

```text
data/NEWS.csv
```

Expected columns include:

- `title`
- `summary`
- `image_url`
- `published`
- `url`
- `images`

The application creates additional fields during runtime, such as processed text, tokens, document IDs, publication date, and detected subject.

## How It Works

1. Load the news dataset from `data/NEWS.csv`
2. Clean summaries and publication dates
3. Detect a broad subject category from article keywords
4. Preprocess text by lowercasing, removing URLs and punctuation, removing stop words, and lemmatizing terms
5. Build an inverted index from document tokens
6. Process the user query
7. Retrieve matching documents
8. Rank results using TF-IDF
9. Apply query expansion and pseudo relevance feedback
10. Display ranked results in the web interface

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/4khello/benha-news-search-engine.git
cd benha-news-search-engine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Evaluation

The project includes a simple evaluation route:

```text
http://127.0.0.1:5000/evaluate
```

It runs several sample queries, measures response time, and displays the top result for each query.

## Example Queries

- `artificial intelligence`
- `government policy`
- `health hospital`
- `football team`
- `business market`
- `school students`

## Notes

- The search engine currently uses classic Information Retrieval methods rather than semantic embeddings.
- Subject detection is rule-based using keyword matching.
- Query expansion uses a small manually defined synonym map.
- The dataset should be reviewed before publishing if it contains copyrighted or private content.

## Future Improvements

- Add BM25 ranking
- Add semantic search using sentence embeddings
- Add pagination for search results
- Add more advanced evaluation metrics such as Precision@K and MAP
- Add an API endpoint for search results
- Improve subject classification with a trained classifier

## CV Summary

**Benha News Search Engine**  
Built a Flask-based news search engine that preprocesses article text, builds an inverted index, and ranks results using TF-IDF. Added query expansion, pseudo relevance feedback, filtering by subject/title/date, and an evaluation page to measure retrieval performance.

**Tech:** Python, Flask, Pandas, NLTK, HTML, CSS
