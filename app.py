import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# -----------------------------
# Basic setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "NEWS.csv")

def download_nltk_data():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package)


try:
    nltk.data.find("tokenizers/punkt")
    USE_NLTK_TOKENIZER = True
except LookupError:
    USE_NLTK_TOKENIZER = False

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "this", "to", "was",
    "we", "were", "will", "with", "you", "your", "but", "not", "they",
    "them", "which", "who", "what", "when", "where", "why", "how", "about",
    "after", "before", "into", "over", "under", "more", "most", "new"
}

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = FALLBACK_STOPWORDS

try:
    nltk.data.find("corpora/wordnet")
    USE_WORDNET = True
except LookupError:
    USE_WORDNET = False

lemmatizer = WordNetLemmatizer()

def safe_lemmatize(word):
    if USE_WORDNET:
        return lemmatizer.lemmatize(word)
    # Very small fallback stem-like normalization for offline machines.
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


# -----------------------------
# Phase 1: Preprocessing
# -----------------------------
def preprocessing(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s]", " ", text)

    if USE_NLTK_TOKENIZER:
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = re.findall(r"[a-z]+", text)
    else:
        tokens = re.findall(r"[a-z]+", text)

    processed = [safe_lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(processed)

def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Dataset loading and preparation
# -----------------------------
def clean_date(value):
    date_value = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(date_value):
        return pd.NaT
    return date_value.tz_convert(None)


def detect_subject(text):
    text = (text or "").lower()
    subject_keywords = {
        "Politics": ["government", "minister", "election", "parliament", "policy", "pm", "president", "council", "law"],
        "Business": ["business", "market", "bank", "economy", "money", "price", "company", "stock", "tax", "royal mail"],
        "Health": ["health", "doctor", "hospital", "disease", "nhs", "mental", "medical", "patient"],
        "Sports": ["sport", "football", "match", "club", "league", "cup", "player", "coach", "team"],
        "Technology": ["technology", "tech", "ai", "software", "app", "online", "cyber", "data", "digital"],
        "Education": ["school", "university", "student", "education", "teacher", "college", "exam"],
        "World": ["world", "war", "country", "international", "global", "china", "russia", "ukraine", "israel", "gaza"],
        "Entertainment": ["film", "music", "actor", "tv", "show", "celebrity", "cinema", "bbc"],
    }
    scores = {subject: sum(1 for keyword in keywords if keyword in text) for subject, keywords in subject_keywords.items()}
    best_subject, best_score = max(scores.items(), key=lambda item: item[1])
    return best_subject if best_score > 0 else "General"


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.fillna("")
    df["summary"] = df["summary"].apply(strip_html)
    if "text" not in df.columns:
        df["text"] = df["title"].astype(str) + " " + df["summary"].astype(str)

    df["docno"] = df.index.astype(str)
    df["published_date"] = df["published"].apply(clean_date)
    df["date_string"] = df["published_date"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    df["subject"] = df["text"].apply(detect_subject)
    df["processed_text"] = df["text"].apply(preprocessing)
    df["tokens"] = df["processed_text"].apply(lambda value: value.split() if isinstance(value, str) else [])
    return df


# -----------------------------
# Inverted index
# -----------------------------
def build_inverted_index(df):
    inverted_index = defaultdict(dict)
    document_lengths = {}

    for _, row in df.iterrows():
        doc_id = int(row["docno"])
        tokens = row["tokens"]
        document_lengths[doc_id] = len(tokens)
        term_counts = Counter(tokens)

        for term, tf in term_counts.items():
            inverted_index[term][doc_id] = tf

    return dict(inverted_index), document_lengths


df = load_dataset()
inverted_index, document_lengths = build_inverted_index(df)
N_DOCUMENTS = len(df)

# -----------------------------
# Statistics and postings
# -----------------------------
def collection_statistics():
    number_of_terms = len(inverted_index)
    number_of_postings = sum(len(postings) for postings in inverted_index.values())
    number_of_tokens = sum(document_lengths.values())
    empty_documents = sum(1 for length in document_lengths.values() if length == 0)

    return {
        "Number of documents": N_DOCUMENTS,
        "Number of terms": number_of_terms,
        "Number of postings": number_of_postings,
        "Number of tokens": number_of_tokens,
        "Empty documents": empty_documents,
    }


def get_term_postings(term):
    processed_terms = preprocessing(term).split()
    if not processed_terms:
        return pd.DataFrame(columns=["DocID", "Term_Frequency", "Doc_Length"])

    term = processed_terms[0]
    postings = inverted_index.get(term, {})
    postings_list = []

    for doc_id, tf in postings.items():
        postings_list.append({
            "DocID": doc_id,
            "Term_Frequency": tf,
            "Doc_Length": document_lengths.get(doc_id, 0),
        })

    return pd.DataFrame(postings_list).sort_values("Term_Frequency", ascending=False)


# Print statistics in the terminal when the project starts.
print("\n================ Phase 1 Collection Statistics ================")
for key, value in collection_statistics().items():
    print(f"{key}: {value}")
print("================================================================\n")


# -----------------------------
# Query expansion
# -----------------------------
SYNONYM_MAP = {
    "ai": ["artificial", "intelligence", "technology", "digital"],
    "artificial": ["ai", "technology"],
    "business": ["market", "company", "economy"],
    "economy": ["business", "market", "money"],
    "health": ["medical", "hospital", "doctor", "nhs"],
    "sport": ["football", "match", "team", "league"],
    "sports": ["football", "match", "team", "league"],
    "politics": ["government", "minister", "election", "parliament"],
    "government": ["minister", "policy", "law"],
    "education": ["school", "student", "university", "exam"],
    "war": ["conflict", "military", "ukraine", "gaza"],
    "climate": ["weather", "environment", "warming", "emissions"],
    "movie": ["film", "cinema", "actor"],
    "film": ["movie", "cinema", "actor"],
}


def expand_with_synonyms(query_terms):
    expanded = set(query_terms)
    for term in query_terms:
        for synonym in SYNONYM_MAP.get(term, []):
            processed_synonyms = preprocessing(synonym).split()
            expanded.update(processed_synonyms)
    return list(expanded)


def idf(term):
    df_term = len(inverted_index.get(term, {}))
    return math.log((N_DOCUMENTS + 1) / (df_term + 1)) + 1


def tf_idf_score(doc_id, query_terms, original_terms=None):
    score = 0.0
    original_terms = set(original_terms or [])

    for term in query_terms:
        tf = inverted_index.get(term, {}).get(doc_id, 0)
        if tf > 0:
            # Log-normalized TF-IDF.
            weight = 1 + math.log(tf)
            term_score = weight * idf(term)

            # Original query terms are more important than expanded terms.
            if term in original_terms:
                term_score *= 2.0
            else:
                term_score *= 0.7

            score += term_score

    return score


def docs_containing_all_terms(terms):
    if not terms:
        return set()

    posting_sets = []
    for term in terms:
        postings = set(inverted_index.get(term, {}).keys())
        if not postings:
            return set()
        posting_sets.append(postings)

    return set.intersection(*posting_sets) if posting_sets else set()


def docs_containing_any_terms(terms):
    docs = set()
    for term in terms:
        docs.update(inverted_index.get(term, {}).keys())
    return docs


def relevance_feedback_terms(top_doc_ids, original_terms, max_terms=5):
    """Pseudo relevance feedback: extract the most frequent useful terms from top-ranked documents."""
    feedback_counter = Counter()
    original_terms = set(original_terms)

    for doc_id in top_doc_ids:
        tokens = df.loc[df["docno"] == str(doc_id), "tokens"]
        if not tokens.empty:
            feedback_counter.update(tokens.iloc[0])

    useful_terms = []
    for term, _ in feedback_counter.most_common(30):
        if term not in original_terms and term in inverted_index and len(term) > 2:
            useful_terms.append(term)
        if len(useful_terms) == max_terms:
            break

    return useful_terms


# -----------------------------
# Search with filters
# -----------------------------
def apply_filters(candidate_doc_ids, title_filter="", subject_filter="", date_from="", date_to=""):
    filtered_ids = []
    title_filter = title_filter.lower().strip()

    from_date = pd.to_datetime(date_from, errors="coerce") if date_from else pd.NaT
    to_date = pd.to_datetime(date_to, errors="coerce") if date_to else pd.NaT

    for doc_id in candidate_doc_ids:
        row = df.iloc[doc_id]

        if title_filter and title_filter not in str(row["title"]).lower():
            continue

        if subject_filter and subject_filter != "All" and row["subject"] != subject_filter:
            continue

        published_date = row["published_date"]
        if not pd.isna(from_date):
            if pd.isna(published_date) or published_date < from_date:
                continue
        if not pd.isna(to_date):
            if pd.isna(published_date) or published_date > to_date:
                continue

        filtered_ids.append(doc_id)

    return filtered_ids


def search_engine(query, title_filter="", subject_filter="All", date_from="", date_to="", top_k=20):
    start_time = time.time()

    original_terms = preprocessing(query).split()
    if not original_terms:
        return {
            "query_terms": [],
            "expanded_terms": [],
            "feedback_terms": [],
            "results": [],
            "total_results": 0,
            "search_time": 0,
            "message": "Please enter a valid English query.",
        }

    # Requirement: retrieve documents that contain all original query terms.
    candidate_docs = docs_containing_all_terms(original_terms)

    # Synonym expansion helps when the strict AND query returns no result.
    expanded_terms = expand_with_synonyms(original_terms)
    if not candidate_docs:
        candidate_docs = docs_containing_any_terms(expanded_terms)

    candidate_docs = apply_filters(candidate_docs, title_filter, subject_filter, date_from, date_to)

    first_ranking = sorted(
        [(doc_id, tf_idf_score(doc_id, expanded_terms, original_terms)) for doc_id in candidate_docs],
        key=lambda item: item[1],
        reverse=True,
    )

    top_feedback_docs = [doc_id for doc_id, _ in first_ranking[:5]]
    feedback_terms = relevance_feedback_terms(top_feedback_docs, original_terms, max_terms=5)
    final_terms = list(dict.fromkeys(expanded_terms + feedback_terms))

    final_ranking = sorted(
        [(doc_id, tf_idf_score(doc_id, final_terms, original_terms)) for doc_id in candidate_docs],
        key=lambda item: item[1],
        reverse=True,
    )

    results = []
    for rank, (doc_id, score) in enumerate(final_ranking[:top_k], start=1):
        row = df.iloc[doc_id]
        image_source = row.get("image_url", "") or row.get("images", "")
        results.append({
            "rank": rank,
            "docno": row["docno"],
            "title": row.get("title", "No title"),
            "summary": row.get("summary", ""),
            "url": row.get("url", ""),
            "image_url": image_source,
            "published": row.get("date_string", "Unknown"),
            "subject": row.get("subject", "General"),
            "score": round(score, 4),
        })

    search_time = round(time.time() - start_time, 4)
    return {
        "query_terms": original_terms,
        "expanded_terms": final_terms,
        "feedback_terms": feedback_terms,
        "results": results,
        "total_results": len(final_ranking),
        "search_time": search_time,
        "message": "",
    }


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_search_engine():
    test_queries = [
        "government policy",
        "health hospital",
        "football team",
        "artificial intelligence",
        "business market",
        "school students",
    ]

    evaluations = []
    total_time = 0

    for query in test_queries:
        output = search_engine(query, top_k=10)
        total_time += output["search_time"]
        evaluations.append({
            "query": query,
            "results_count": output["total_results"],
            "top_result": output["results"][0]["title"] if output["results"] else "No results",
            "time": output["search_time"],
        })

    average_time = round(total_time / len(test_queries), 4)
    return evaluations, average_time


# -----------------------------
# Flask UI
# -----------------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    query = request.args.get("q", "")
    title_filter = request.args.get("title", "")
    subject_filter = request.args.get("subject", "All")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    subjects = ["All"] + sorted(df["subject"].dropna().unique().tolist())
    stats = collection_statistics()

    search_output = None

    if query.strip():
        search_output = search_engine(
            query,
            title_filter,
            subject_filter,
            date_from,
            date_to
        )

    return render_template(
        "index.html",
        query=query,
        title_filter=title_filter,
        subject_filter=subject_filter,
        date_from=date_from,
        date_to=date_to,
        subjects=subjects,
        stats=stats,
        search_output=search_output,
        page="search",
    )



@app.route("/evaluate", methods=["GET"])
def evaluate():
    subjects = ["All"] + sorted(df["subject"].dropna().unique().tolist())
    stats = collection_statistics()
    evaluations, average_time = evaluate_search_engine()

    return render_template(
        "index.html",
        subjects=subjects,
        stats=stats,
        evaluations=evaluations,
        average_time=average_time,
        search_output=None,
        page="evaluate",
        query="",
        title_filter="",
        subject_filter="All",
        date_from="",
        date_to="",
    )


print("Registered routes:")
print(app.url_map)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)