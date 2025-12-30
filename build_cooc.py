import io
import os
from collections import Counter

# ---------------------------------------------------------
# PROJECT CONFIGURATION AND FILE PATHS
# ---------------------------------------------------------

# Automatically get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
corpus_path = os.path.join(BASE_DIR, "wiki_cleaned_final.txt")
vocab_path = os.path.join(BASE_DIR, "vocab.txt")
cooc_path = os.path.join(BASE_DIR, "cooc.txt") 

# Stopwords (Must be consistent with other scripts)
CUSTOM_STOPWORDS = {
    "ve", "bir", "bu", "şu", "o",
    "ki", "ile", "için", "olarak", "en", "sonra",
    "da", "de", "li",
    "ol", "et", "yap", "al"
}

# Window size: check 5 words left and right
WINDOW_SIZE = 5

# ---------------------------------------------------------
# STEP 1: LOAD VOCABULARY
# ---------------------------------------------------------

word2id = {}

print(f"Loading vocabulary: {vocab_path}")

try:
    with io.open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.strip()
            if word:
                word2id[word] = idx
except FileNotFoundError:
    print(f"\n[ERROR] 'vocab.txt' not found!")
    print("Please run 'build_vocab.py' first.")
    exit()

print(f"Vocabulary size: {len(word2id)} words loaded.")

# ---------------------------------------------------------
# STEP 2: CALCULATE CO-OCCURRENCE MATRIX
# ---------------------------------------------------------

cooc = Counter()

print(f"Scanning corpus and counting relationships: {corpus_path}")

try:
    with io.open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            
            tokens = line.strip().split()
            if not tokens:
                continue

            # Filter out stopwords and unknown words
            filtered_tokens = []
            for t in tokens:
                if t not in CUSTOM_STOPWORDS and t in word2id:
                    filtered_tokens.append(t)
            
            tokens = filtered_tokens
            
            # If line too short, skip
            if len(tokens) < 2:
                continue

            # Word → ID conversion
            ids = [word2id[t] for t in tokens]

            # Sliding window algorithm
            for center_pos, w_i in enumerate(ids):
                left = max(0, center_pos - WINDOW_SIZE)
                right = min(len(ids), center_pos + WINDOW_SIZE + 1)

                for ctx_pos in range(left, right):
                    if ctx_pos == center_pos:
                        continue

                    w_j = ids[ctx_pos]

                    # Symmetric relationship
                    if w_i < w_j:
                        pair = (w_i, w_j)
                    else:
                        pair = (w_j, w_i)

                    cooc[pair] += 1

            if line_idx % 5000 == 0:
                print(f"-> {line_idx} lines processed...")

except FileNotFoundError:
    print(f"\n[ERROR] Corpus file not found: {corpus_path}")
    exit()

print(f"Processing completed. Total unique co-occurrence pairs: {len(cooc)}")

# ---------------------------------------------------------
# STEP 3: SAVE RESULTS
# ---------------------------------------------------------

print(f"Saving co-occurrence matrix: {cooc_path}")

with io.open(cooc_path, "w", encoding="utf-8") as out:
    for (i_id, j_id), count in cooc.items():
        out.write(f"{i_id}\t{j_id}\t{count}\n")

print("Co-occurrence matrix successfully created.")
