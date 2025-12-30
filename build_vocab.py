import io
import os
from collections import Counter

# ---------------------------------------------------------
# PROJECT CONFIGURATION AND FILE PATHS
# ---------------------------------------------------------

# Automatically get the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: Cleaned corpus file
corpus_path = os.path.join(BASE_DIR, "wiki_cleaned_final.txt")

# Outputs: Vocabulary list and vocabulary with frequencies
vocab_path = os.path.join(BASE_DIR, "vocab.txt")
vocab_freq_path = os.path.join(BASE_DIR, "vocab_freq.txt")

# ---------------------------------------------------------
# PARAMETERS AND SETTINGS
# ---------------------------------------------------------

# Stopwords:
# We exclude function words and helpers that carry low semantic meaning.
CUSTOM_STOPWORDS = {
    "ve", "bir", "bu", "şu", "o",
    "ki", "ile", "için", "olarak", "en", "sonra",
    "da", "de", "li",
    "ol", "et", "yap", "al"
}

# Minimum Frequency Threshold (Min Count):
# Words occurring less than this threshold (rare words, typos, etc.)
# are removed to improve the model's stability.
MIN_COUNT = 30

# ---------------------------------------------------------
# STEP 1: COUNT WORD FREQUENCIES
# ---------------------------------------------------------

# Counter object to hold word frequencies
freq = Counter()

print(f"Reading data and calculating frequencies: {corpus_path}")

try:
    with io.open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):

            tokens = line.strip().split()
            if not tokens:
                continue

            # Filter out stopwords
            tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]

            # Update frequency counter
            freq.update(tokens)

            # Progress update
            if i % 5000 == 0:
                print(f"-> {i} lines processed...")

except FileNotFoundError:
    print("\n[CRITICAL ERROR] File not found!")
    print(f"Missing path: {corpus_path}")
    print("Please ensure 'wiki_cleaned_final.txt' is located in the same directory as this script.")
    exit()

print(f"Raw vocabulary size (before filtering): {len(freq)}")

# ---------------------------------------------------------
# STEP 2: FILTERING AND SORTING
# ---------------------------------------------------------

print(f"Removing words with frequency below {MIN_COUNT}...")

# Filter by minimum count
filtered_items = [(w, c) for (w, c) in freq.items() if c >= MIN_COUNT]

# Sort by descending frequency
filtered_items.sort(key=lambda x: x[1], reverse=True)

final_vocab_size = len(filtered_items)
print(f"Final vocabulary size: {final_vocab_size} words")

# ---------------------------------------------------------
# STEP 3: SAVE OUTPUT FILES
# ---------------------------------------------------------

print("Saving output files...")

# 1. vocab.txt: Only the list of words (required for training)
with io.open(vocab_path, "w", encoding="utf-8") as vout:
    for w, _ in filtered_items:
        vout.write(w + "\n")

# 2. vocab_freq.txt: Word + count (useful for analysis/debugging)
with io.open(vocab_freq_path, "w", encoding="utf-8") as fout:
    for w, c in filtered_items:
        fout.write(f"{w}\t{c}\n")

print("-" * 30)
print("PROCESS COMPLETED SUCCESSFULLY")
print(f"Generated files:\n1. {vocab_path}\n2. {vocab_freq_path}")
print("-" * 30)
