import io
import os
import re

# ---------------------------------------------------------
# PROJECT CONFIGURATION AND FILE PATHS
# ---------------------------------------------------------

# Detect the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: Lemmatized raw text
input_path = os.path.join(BASE_DIR, "wiki_lemma_tokens.txt")

# Output: Fully cleaned text for model training
output_path = os.path.join(BASE_DIR, "wiki_cleaned_final.txt")

# ---------------------------------------------------------
# STEP 1: DEFINE CLEANING RULES (REGEX)
# ---------------------------------------------------------

# Regex definition:
# Purpose: keep only alphabetic characters (including Turkish letters).
# [^...] means "everything except these".
# So this finds everything that is NOT a-z, A-Z or Turkish letters (çğıöşü...).
clean_re = re.compile(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]+")

# ---------------------------------------------------------
# STEP 2: MAIN TEXT PROCESSING LOOP
# ---------------------------------------------------------

print(f"Starting cleaning process: {input_path}")

try:
    with io.open(input_path, "r", encoding="utf-8") as fin, \
         io.open(output_path, "w", encoding="utf-8") as fout:

        lines_written = 0
        
        for i, line in enumerate(fin, start=1):
            
            # 1. Normalize: convert to lowercase
            # This ensures "Apple" and "apple" are treated as the same token.
            line = line.strip().lower()

            # 2. Noise removal: Replace non-letter characters with space
            line = clean_re.sub(" ", line)

            # 3. Tokenization: split into words
            tokens = line.split()

            # 4. Filtering: remove single-character tokens
            # This improves vocabulary quality (removes noise like "x", "v", etc.)
            tokens = [t for t in tokens if len(t) > 1]

            # If the line becomes empty after cleaning, skip it
            if not tokens:
                continue

            # Write cleaned line to output file
            fout.write(" ".join(tokens) + "\n")
            lines_written += 1

            # Show progress every 5000 lines
            if i % 5000 == 0:
                print(f"-> {i} lines processed...")

    print("-" * 30)
    print("CLEANING PROCESS COMPLETED")
    print(f"Total cleaned lines written: {lines_written}")
    print(f"Output file: {output_path}")
    print("-" * 30)

except FileNotFoundError:
    print("\n[ERROR] Input file not found!")
    print(f"Missing file: {input_path}")
    print("Please make sure the file name and directory are correct.")
