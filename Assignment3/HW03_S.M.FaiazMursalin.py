import nltk as tk
import concurrent  # imported for using multithreading features for reading documents in the directory
import os, math
import numpy as np
from collections import Counter
from itertools import combinations
import pandas as pd

'''Function to display course and student information'''


def display_information():
    print("=================== CSC790-IR Homework 03 ==============")
    print("First Name: S.M. Faiaz")
    print("Last Name: Mursalin")
    print("========================================================")


'''The below Function utilizes Multiprocessing to read the documents in the document directory and calls
 the process_text_file function every time'''


def read_documents_using_multithreading(directory, filepaths, stopwords, punctuation_special_character_file_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        documents = list(
            executor.map(lambda file: process_text_file(os.path.join(directory, file), stopwords,
                                                        punctuation_special_character_file_path),
                         filepaths))
    return documents


''' Function to read the stop words from the stopword.txt file and tokenize them and return the list of stop words'''


def read_stop_words(stop_words_file):
    with open(stop_words_file, "r") as file:
        content = file.read()
        stop_word_tokens = tk.word_tokenize(content)
    return stop_word_tokens


''' Function to read the punctuations and special characters and tokenize them and return the list of 
    special character and punctuations'''


def read_punctuations_special_char(punctuation_special_character_file_path):
    # Read Punctuations and special character from a file
    with open(punctuation_special_character_file_path, "r") as file:
        content = file.read()
        punctuation_special_char_tokens = tk.word_tokenize(content)
    punctuation_special_char_tokens.append("..")
    return punctuation_special_char_tokens


'''process each query text or input text '''


def process_text_file(filepath, stopwords, punctuations_special_char):
    with open(filepath, 'r') as file:
        text = file.read()
        stemmed_tokens = process_text(text, stopwords, punctuations_special_char)
    return stemmed_tokens


''' Tokenize and preprocess the text'''


def process_text(text, stopwords, punctuations_special_char):
    stemmer = tk.stem.PorterStemmer()
    # Tokenize
    tokens = tk.word_tokenize(text)
    # lower case all the tokens
    tokens = [token.lower() for token in tokens]
    # remove punctuations
    tokens = [token for token in tokens if token not in punctuations_special_char]
    # remove stop words
    tokens = [token for token in tokens if token not in stopwords]
    # stemming the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


''' Function to compute sublinear tf scaling'''


def compute_sublinear_tf_scaling(tf):
    if tf > 0:
        return 1 + math.log10(tf)
    else:
        return 0


'''Function to compute cosine similarity between two vectors'''


def compute_similarity(vec1, vec2):
    # Compute cosine similarity between two documents
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine_sim


''' Function to compute scores for each document pair'''
def compute_scores(vector_space):
    scores = {'file1': [], 'file2': [], 'similarity': []}
    pairs = combinations(vector_space.columns.tolist(), 2)
    for pair in pairs:
        scores['file1'].append(pair[0])
        scores['file2'].append(pair[1])
        scores['similarity'].append(
            compute_similarity(vector_space[pair[0]].values, vector_space[pair[1]].values))

    scores = pd.DataFrame.from_dict(scores)
    return scores.sort_values(by='similarity', ascending=False)


''' Function to display results'''
def display_results(K):
    print("Calculating scores")
    vector_spaces = [tf_vector_space, tf_idf_vector, wf_idf_vector]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        scores = executor.map(compute_scores, vector_spaces)

    print(f"The number of unique words is: {N_TOKENS}")
    print(f"The top {K} most frequent words are:")
    for i, token in enumerate(collection_frequency.head(K).index):
        print(f"{i + 1:5d}. {token}")
    measures = ["tf", "tf-idf", "sublinear scaling"]
    print(f"The top {K} closest documents are: ")
    for i, measure, score in zip(range(len(measures)), measures, scores):
        print(f"{i + 1:5d}. Using {measure}")
        for _, row in score.head(K).iterrows():
            print(f"\t{row['file1']}, {row['file2']} with  a similarity of {np.round(row['similarity'], 2)}")


if __name__ == '__main__':
    # display course and students information of myself and what course
    display_information()
    # relative document path
    documents_path = "./documents/"
    # relative stopword.txt file path
    stop_words_file_path = "./stopwords.txt"
    # Read stop words from stopwords.txt so that we can get rid of all the stop words after performing stemming
    stop_words = read_stop_words(stop_words_file_path)
    # list of files in the documents so that we can keep the file name in the posting not the document id this helps
    # better to understand
    files = [filename for filename in os.listdir(documents_path) if filename.endswith(".txt")]
    # relative special-char.tx containing punctuation and special chars file path
    punctuation_special_character_file_path = "./special_chars-queries/special-chars.txt"
    # Read punctuation and special chars
    punctuations_special_char = read_punctuations_special_char(punctuation_special_character_file_path)
    # Read the documents using multithreading
    documents = read_documents_using_multithreading(documents_path, files, stop_words, punctuations_special_char)

    # Compute TF vector space
    tf_vector_space = {}
    for filename, document in zip(files, documents):
        file_id = filename.split(os.path.sep)[-1].split(".txt")[0]
        tf_vector_space[file_id] = Counter(document)

    tf_vector_space = pd.DataFrame.from_dict(tf_vector_space)
    tf_vector_space = tf_vector_space.fillna(0)

    N_TOKENS, N_DOCUMENTS = tf_vector_space.shape

    # Document frequency calculation
    document_frequency = tf_vector_space.astype(bool).sum(axis=1)

    # Collection frequency calculation
    collection_frequency = tf_vector_space.sum(axis=1).sort_values(ascending=False)

    # Inverse document frequency calculation
    inverse_document_frequency = np.log10(N_DOCUMENTS / document_frequency)

    # TF-IDF vector calculation
    tf_idf_vector = tf_vector_space.mul(inverse_document_frequency, axis=0)

    # WF calculation
    wf = tf_vector_space.applymap(compute_sublinear_tf_scaling)
    wf_idf_vector = wf.mul(inverse_document_frequency, axis=0)

    # Display the results
    display_results(K=10)
