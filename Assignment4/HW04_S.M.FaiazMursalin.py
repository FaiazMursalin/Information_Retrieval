import nltk as tk
import concurrent.futures as cf
import os
from collections import Counter

import numpy as np
import pandas as pd

# Function to display course and student information
def display_information():
    print("=================== CSC790-IR Homework 04 ==============")
    print("First Name: S.M. Faiaz")
    print("Last Name: Mursalin")
    print("========================================================")

# Function to read the stop words from the stopword.txt file and tokenize them and return the list of stop words
def read_stop_words(stop_words_file):
    with open(stop_words_file, "r") as file:
        return file.read().split()

# Function to read the punctuations and special characters and tokenize them and return the list of special character and punctuations
def read_punctuations_special_char(punctuation_special_character_file_path):
    with open(punctuation_special_character_file_path, "r") as file:
        return file.read().split() + [".."]

# The below Function utilizes Multiprocessing to read the documents in the document directory and calls
# the process_text_file function every time
def read_documents_using_multithreading(directory, filepaths, stopwords, punctuation_special_character_file_path):
    vector = {}
    with cf.ThreadPoolExecutor() as executor:
        documents = list(
            executor.map(lambda file: process_text_file(os.path.join(directory, file), stopwords,
                                                         punctuation_special_character_file_path),
                         filepaths))
    for doc_id, counts in documents:
        for token, count in counts.items():
            if token in vector:
                vector[token][doc_id] = count
            else:
                vector[token] = {doc_id: count}

    df = pd.DataFrame.from_dict(vector, orient='index').fillna(0)
    return df.copy(), df.shape

# Function to process a text file
def process_text_file(filepath, stopwords, punctuations_special_char):
    doc_id = os.path.basename(filepath).split(".txt")[0]
    with open(filepath, 'r') as file:
        text = file.read()
        stemmed_tokens = process_text(text, stopwords, punctuations_special_char)
    return doc_id, dict(Counter(stemmed_tokens))

# Function to calculate ct value
def calculate_ct(N, df_t, S, s):
    return np.log(((s + 0.5) / (S - s + 0.5)) / ((df_t - s + 0.5) / (N - df_t - S + s + 0.5)))

# Function to compute ct for each term
def compute_ct_for_term(args):
    term, n_docs, doc_freq, relevance, vector_space = args
    merged_df = relevance.merge(
        vector_space.loc[term], left_index=True, right_index=True, how='left').fillna(0)
    relevant_and_present = merged_df[
        (merged_df['relevance'] == 1) & (merged_df[term] > 0)]
    return term, calculate_ct(n_docs, doc_freq[term], relevance['relevance'].sum(),
                               relevant_and_present.shape[0])

# Tokenize and preprocess the text
def process_text(text, stopwords, punctuations_special_char):
    stemmer = tk.stem.PorterStemmer()
    # Tokenize
    tokens = tk.word_tokenize(text.lower())
    # Remove punctuations and stop words, and stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens if
                      token not in punctuations_special_char and token not in stopwords]
    return stemmed_tokens

if __name__ == '__main__':
    # Display course and student information
    display_information()

    # Relative document path
    documents_path = "./documents/"
    # Relative stopword.txt file path
    stop_words_file_path = "./stopwords.txt"
    # Read stop words from stopwords.txt so that we can get rid of all the stop words after performing stemming
    stop_words = set(read_stop_words(stop_words_file_path))
    # Relative special-char.tx containing punctuation and special chars file path
    punctuation_special_character_file_path = "./special-chars.txt"
    # Read punctuation and special chars
    punctuations_special_char = set(read_punctuations_special_char(punctuation_special_character_file_path))

    # List of files in the documents path
    files = [filename for filename in os.listdir(documents_path) if filename.endswith(".txt")]

    # Read the documents using multithreading
    documents, (n_tokens, n_docs) = read_documents_using_multithreading(documents_path, files, stop_words,
                                                                        punctuations_special_char)

    # Query filepath
    query_path = "./query.txt"

    # Read query terms
    with open(query_path, "r") as q_in:
        terms = process_text(q_in.read(), stop_words, punctuations_special_char)

    # Read relevance labels
    relevance = pd.read_csv("file_label.txt", index_col=0, header=None, names=['doc_id', 'relevance'])

    # Calculate presence matrix and document frequency
    presence_df = (documents > 0).astype(int)
    doc_freq = documents.ne(0).sum(axis=1)

    args_list = [(term, n_docs, doc_freq, relevance, documents)
                 for term in terms]
    # for each term in the query, compute ct
    with cf.ProcessPoolExecutor() as executor:
        ct_results = list(executor.map(compute_ct_for_term, args_list))

    ct = pd.Series({term: value for term, value in ct_results})

    # for each document, if term present, add ct
    filtered_doc = presence_df.loc[ct.index]
    rsv = filtered_doc.multiply(ct, axis=0).sum()
    rsv.name = "ct"

    # Print the top 10 results
    output_df = relevance.join(rsv).sort_values(by="ct", ascending=False)
    top_results = output_df.head(10)
    for idx, row in top_results.iterrows():
        print(
            f"RSV {{ {idx:<10} }} = {row['ct']:>6.2f} {int(row['relevance']):>3d}")
