import io
import os

# ---------------------------------------------------------
# PROJECT CONFIGURATION AND FILE PATHS
# ---------------------------------------------------------

# Make file paths dynamic by detecting the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: Raw Wikipedia dump (typically contains XML-like tags)
input_path = os.path.join(BASE_DIR, "raw_wiki.txt")

# Output: Text cleaned from XML/meta tags
output_path = os.path.join(BASE_DIR, "wiki_clean.txt")

# ---------------------------------------------------------
# TEXT PREPROCESSING AND XML CLEANING
# ---------------------------------------------------------

print(f"Starting raw data cleaning: {input_path}")

try:
    with io.open(input_path, "r", encoding="utf-8") as fin, \
         io.open(output_path, "w", encoding="utf-8") as fout:
        
        lines_written = 0
        
        for i, line in enumerate(fin, start=1):
            
            # 1. XML/Meta Tag Removal:
            # Wikipedia dump files often include <doc id="..."> and </doc> tags.
            # These are not natural language and harm model training.
            if line.startswith("<doc ") or line.startswith("</doc>"):
                continue

            # 2. Whitespace Cleanup:
            # Remove unnecessary whitespace from the beginning and end.
            line = line.strip()

            # 3. Empty Line Filter:
            # Skip the line if it becomes empty after stripping.
            if not line:
                continue

            # Write cleaned line to output file
            fout.write(line + "\n")
            lines_written += 1

            # Progress info for large files
            if i % 100000 == 0:
                print(f"-> {i} lines processed...")

    print("-" * 30)
    print("RAW DATA CLEANING COMPLETED")
    print(f"Total cleaned lines written: {lines_written}")
    print(f"Output file: {output_path}")
    print("-" * 30)

except FileNotFoundError:
    print("\n[ERROR] Input file not found!")
    print(f"Missing file: {input_path}")
    print("Please ensure 'raw_wiki.txt' is in the same directory as this script.")
