import numpy as np
import sys
import os
import matplotlib

# --- GRAPHIC BACKEND CONFIGURATION ---
# Use TkAgg to ensure the plot window opens correctly on all systems
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import norm

# --- DYNAMIC PATH CONFIGURATION ---
# Automatically determine the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths dynamically
vector_file_path = os.path.join(BASE_DIR, "vectors.txt")
vocab_file_path = os.path.join(BASE_DIR, "vocab.txt")
cooc_file_path = os.path.join(BASE_DIR, "cooc.txt")


def load_vectors(vector_file):
    """Loads trained word vectors from the file into a dictionary."""
    print(f"Loading vectors from: {vector_file}...")
    embeddings = {}

    if not os.path.exists(vector_file):
        print("ERROR: vectors.txt not found. Please run train_glove.py first.")
        return None

    with open(vector_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word] = vector

    print(f"Total vectors loaded: {len(embeddings)}")
    return embeddings


def load_prediction_data(vocab_file, cooc_file):
    """
    Loads vocabulary mapping and co-occurrence data for the 'Next Word Prediction' feature.
    Returns: word2id, id2word, cooc_data
    """
    print("Loading vocabulary and co-occurrence data for prediction module...")

    if not os.path.exists(vocab_file) or not os.path.exists(cooc_file):
        print("WARNING: vocab.txt or cooc.txt missing — prediction module will be disabled.")
        return None, None, None

    # Load Vocabulary
    word2id = {}
    id2word = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word = line.strip().split()[0]
            word2id[word] = i
            id2word[i] = word

    # Load Co-occurrence Data
    cooc_data = {}
    with open(cooc_file, 'r', encoding='utf-8') as f:
        for line in f:
            u, v, c = line.strip().split()
            u, v, c = int(u), int(v), float(c)

            if u not in cooc_data: cooc_data[u] = []
            if v not in cooc_data: cooc_data[v] = []

            cooc_data[u].append((v, c))
            cooc_data[v].append((u, c))

    print("Prediction data loaded successfully.")
    return word2id, id2word, cooc_data


def predict_next_words(word, word2id, id2word, cooc_data, top_n=5):
    """Returns the most frequently co-occurring words for a given word."""
    if word not in word2id or cooc_data is None:
        return []

    word_id = word2id[word]
    if word_id not in cooc_data:
        return []

    neighbors = cooc_data[word_id]
    neighbors.sort(key=lambda x: x[1], reverse=True)

    results = []
    for nid, count in neighbors[:top_n]:
        if nid in id2word:
            results.append((id2word[nid], count))

    return results


def cosine_similarity(v1, v2):
    """Computes cosine similarity between two vectors."""
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


def plot_results(target_word, similar_words, embeddings, title="Word Map"):
    """Visualizes the target word and its neighbors using PCA (2D)."""

    words_to_plot = [target_word] + [w for w, s in similar_words]
    vectors = [embeddings[w] for w in words_to_plot if w in embeddings]

    if len(vectors) < 2:
        print("Not enough data to plot.")
        return

    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))

    # Target Word
    plt.scatter(vectors_2d[0, 0], vectors_2d[0, 1], c='red', s=150, edgecolors='black')
    plt.annotate(words_to_plot[0], (vectors_2d[0, 0], vectors_2d[0, 1]),
                 xytext=(5, 5), textcoords='offset points', color='red', weight='bold')

    # Neighbors
    plt.scatter(vectors_2d[1:, 0], vectors_2d[1:, 1], c='blue', s=80, alpha=0.6)
    for i in range(1, len(words_to_plot)):
        plt.annotate(words_to_plot[i], (vectors_2d[i, 0], vectors_2d[i, 1]),
                     xytext=(5, 5), textcoords='offset points')

    plt.title(title)
    plt.grid(True)
    plt.legend(["Center Word", "Similar Words"])

    try:
        plt.savefig("last_query.png")
        print("[INFO] Plot saved as last_query.png")
    except Exception as e:
        print(f"Save error: {e}")

    print("Opening visualization window...")
    plt.show()
    plt.close()


def process_query(embeddings, query, word2id, id2word, cooc_data):
    """
    Handles:
    1. Word analogies:   king - man + woman
    2. Similarity search
    3. Context prediction (co-occurrence based)
    """

    # ANALOGY MODE
    if '-' in query and '+' in query:
        try:
            parts = query.replace('+', ' ').replace('-', ' ').split()
            if len(parts) != 3:
                print("Invalid format. Example: king - man + woman")
                return

            w1, w2, w3 = parts
            if w1 not in embeddings or w2 not in embeddings or w3 not in embeddings:
                print("One of the words does not exist in the vocabulary.")
                return

            print(f"\nComputing analogy: {w1} - {w2} + {w3}")
            target_vec = embeddings[w1] - embeddings[w2] + embeddings[w3]

            similarities = []
            for word, vec in embeddings.items():
                if word in {w1, w2, w3}:
                    continue
                similarities.append((word, cosine_similarity(target_vec, vec)))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:10]

            print("-" * 40)
            print("ANALOGY RESULTS:")
            for word, score in top_matches:
                print(f"{word:<20} : {score:.4f}")
            print("-" * 40)

            plot_results(top_matches[0][0], top_matches[1:], embeddings,
                         title=f"Analogy: {query}")

        except Exception as e:
            print(f"ERROR: {e}")

    # SIMILARITY + PREDICTION MODE
    else:
        if query not in embeddings:
            print(f"'{query}' not found in vocabulary.")
            return

        target_vec = embeddings[query]

        # 1. Semantic Similarity
        similarities = [(w, cosine_similarity(target_vec, vec))
                        for w, vec in embeddings.items() if w != query]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:10]

        print("-" * 50)
        print("1. SEMANTIC SIMILARITY (Closest vectors):")
        for w, score in top_matches:
            print(f"   {w:<20} : {score:.4f}")

        # 2. Context Prediction (Co-occurrence)
        print("-" * 50)
        print("2. CONTEXT PREDICTION (Most frequent co-occurrences):")
        predictions = predict_next_words(query, word2id, id2word, cooc_data)
        if predictions:
            for w, count in predictions:
                print(f"   {w:<20} : {int(count)} times")
        else:
            print("   No co-occurrence data available.")
        print("-" * 50)

        plot_results(query, top_matches, embeddings,
                     title=f"'{query}' and Similar Neighbors")


if __name__ == "__main__":
    embeddings = load_vectors(vector_file_path)

    # Load auxiliary data for prediction module
    word2id, id2word, cooc_data = load_prediction_data(vocab_file_path, cooc_file_path)

    if embeddings:
        print("\n--- GLOVE SEARCH & PREDICTION ENGINE ---")
        print("1. Enter a word (e.g., apple)  → Similar words + Prediction")
        print("2. Analogy query (e.g., king - man + woman)")
        print("3. Type 'q' to quit")

        while True:
            query = input("\nQuery: ").strip().lower()
            if query == 'q':
                break
            if query:
                process_query(embeddings, query, word2id, id2word, cooc_data)
