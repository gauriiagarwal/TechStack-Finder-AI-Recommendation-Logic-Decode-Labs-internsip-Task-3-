"""
Project 3 - Tech Stack Recommender (Content-Based Filtering)
Pipeline: Ingestion -> Scoring (TF-IDF + Cosine) -> Sorting -> Filtering (Top-N)
Pure Python: no external libraries needed.
"""
import csv
import math
import sys

CSV_FILE = "raw_skills.csv"
TOP_N = 3
MIN_INPUTS = 3

# Shared vocabulary: map user spellings to the exact names used in the dataset
ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "cloud computing": "cloud",
    "k8s": "kubernetes",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "js": "javascript",
    "nodejs": "node.js",
    "node": "node.js",
    "frontend development": "web design",
    "frontend": "web design",
    "api": "apis",
    "ds": "data structures",
    "dsa": "data structures",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "amazon web services": "aws",
}


def normalize(skill):
    s = skill.strip().lower()
    return ALIASES.get(s, s)


# ---------- Load dataset ----------
def load_items(path):
    items = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items[row["role"]] = [normalize(s) for s in row["skills"].split(";")]
    return items


# ---------- Vector mapping + TF-IDF ----------
def build_vocab(items):
    return sorted({s for skills in items.values() for s in skills})


def compute_idf(items, vocab):
    n_docs = len(items)
    idf = {}
    for term in vocab:
        df = sum(1 for skills in items.values() if term in skills)
        idf[term] = math.log(n_docs / df)  # IDF = log(total docs / docs with term)
    return idf


def tfidf_vector(terms, vocab, idf):
    total = len(terms)
    vec = []
    for term in vocab:
        tf = terms.count(term) / total if total else 0.0  # TF = count / total terms
        vec.append(tf * idf[term])
    return vec


# ---------- Cosine similarity ----------
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:  # zero vector -> cold start, avoid divide by zero
        return 0.0
    return dot / (na * nb)


# ---------- Cold-start fallback ----------
def trending_fallback(items, vocab, idf):
    """If the user gives no usable skills, show roles with the broadest skill sets."""
    ranked = sorted(items, key=lambda r: len(items[r]), reverse=True)
    return [(r, 0.0) for r in ranked[:TOP_N]]


# ---------- Pipeline ----------
def recommend(user_skills, items, vocab, idf):
    # Step 1: Ingestion (clean + keep only skills the system knows)
    cleaned = [normalize(s) for s in user_skills if s.strip()]
    known = [s for s in cleaned if s in vocab]
    unknown = [s for s in cleaned if s not in vocab]
    if not known:
        return trending_fallback(items, vocab, idf), known, unknown

    user_vec = tfidf_vector(known, vocab, idf)

    # Step 2: Scoring
    scores = []
    for role, skills in items.items():
        role_vec = tfidf_vector(skills, vocab, idf)
        scores.append((role, cosine(user_vec, role_vec)))

    # Step 3: Sorting (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Filtering (Top-N)
    return scores[:TOP_N], known, unknown


def main():
    items = load_items(CSV_FILE)
    vocab = build_vocab(items)
    idf = compute_idf(items, vocab)

    print("=== Tech Stack Recommender ===")
    print("Skills I understand:")
    print(", ".join(vocab))
    print()

    # Input: from command line (python recommender.py python sql aws) or interactive
    if len(sys.argv) > 1:
        user_skills = sys.argv[1:]
    else:
        user_skills = []
        while len(user_skills) < MIN_INPUTS:
            s = input(f"Enter skill {len(user_skills) + 1} of {MIN_INPUTS}: ").strip()
            if s:
                user_skills.append(s)
        more = input("Add more skills? (comma separated, or press Enter): ").strip()
        if more:
            user_skills += [x for x in more.split(",") if x.strip()]

    if len(user_skills) < MIN_INPUTS:
        print(f"Please provide at least {MIN_INPUTS} skills.")
        return

    top, known, unknown = recommend(user_skills, items, vocab, idf)

    print("\nYour profile :", ", ".join(known) if known else "(none recognised)")
    if unknown:
        print("Not in dataset (ignored):", ", ".join(unknown))
    if not known:
        print("\nNo known skills -> showing broad-skill roles (cold-start fallback):")
    print(f"\nTop {TOP_N} career matches:")
    for i, (role, score) in enumerate(top, 1):
        print(f"  {i}. {role:<28} match: {score * 100:5.1f}%")
        overlap = [s for s in items[role] if s in known]
        if overlap:
            print(f"     matched skills: {', '.join(overlap)}")


if __name__ == "__main__":
    main()
