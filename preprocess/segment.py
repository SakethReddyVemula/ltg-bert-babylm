# import nltk
# nltk.download('punkt_tab')
# import sys


# input_path = f"../data/processed_{sys.argv[1]}/all.txt"
# output_path = f"../data/processed_{sys.argv[1]}/segmented.txt"

# with open(output_path, "w") as f:
#     for line in open(input_path):
#         line = line.strip() 

#         if len(line) == 0:
#             f.write('\n')
#             continue

#         sentences = nltk.sent_tokenize(line)
#         sentences = '\n'.join(sentences) 
#         f.write(f"{sentences}[PAR]\n")

# Faster segmentation into sentences per line
import nltk
from nltk.tokenize import sent_tokenize
import sys

nltk.download('punkt')

input_path = f"../data/processed_{sys.argv[1]}/all.txt"
output_path = f"../data/processed_{sys.argv[1]}/segmented.txt"

# Read the entire corpus at once
with open(input_path, "r") as infile:
    corpus = infile.read()

# Tokenize the entire corpus
sentences = sent_tokenize(corpus)

# Write the segmented sentences to the output file with '[PAR]' after each sentence
with open(output_path, "w") as outfile:
    for sentence in sentences:
        outfile.write(f"{sentence}[PAR]\n")



