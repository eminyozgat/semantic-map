import io
import os
import stanza

# Download the model on first use (Uncomment the line below if you haven't downloaded it yet)
# stanza.download("tr")

# Initialize Stanza pipeline for Turkish
# Processors: Tokenize, Multi-word Token (MWT), Lemmatization
nlp = stanza.Pipeline(lang="tr", processors="tokenize,mwt,lemma", tokenize_no_ssplit=True)

# Set file paths dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "wiki_clean.txt")       # Input file
output_path = os.path.join(BASE_DIR, "wiki_lemma_tokens.txt") # Output file

print(f"Process started: Reading {input_path}...")

with io.open(input_path, "r", encoding="utf-8") as fin, \
     io.open(output_path, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        # Process the line using the NLP model
        doc = nlp(line)

        lemmas = []
        for sentence in doc.sentences:
            for word in sentence.words:
                lemma = word.lemma

                # Use the original token text if lemma is None
                if lemma is None:
                    lemma = word.text

                # Convert to string if it's not already (safety check)
                lemma = str(lemma)

                lemmas.append(lemma)

        # Write the processed lemmas to the output file
        fout.write(" ".join(lemmas) + "\n")

        # Print progress every 5000 lines
        if i % 5000 == 0:
            print(f"{i} lines processed...")

print("Process completed successfully.")