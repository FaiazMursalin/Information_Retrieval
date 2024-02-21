# if you don't have nltk setup then for the first time while you are setting up you need to do a nltk.download('punkt')
import nltk as tk
import os
import concurrent  # imported for using multithreading features for reading documents in the directory
from collections import defaultdict  # imported for using default dictionary
import json  # imported for saving and loading the inverted index

'''Function to display course and student information'''


def display_information():
    print("=================== CSC790-IR Homework 01 Modified ==============")
    print("First Name: S.M. Faiaz")
    print("Last Name: Mursalin")
    print("========================================================")


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
    return punctuation_special_char_tokens


''' Function which does the following:
    a. reads a document
    b. tokenizes the entire document
    c. lowercase for every token
    d. remove punctuations 
    e. remove stop words
    f. perform stemming'''


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


'''process each query text or input text '''


def process_text_file(filepath, stopwords, punctuations_special_char):
    with open(filepath, 'r') as file:
        text = file.read()
        stemmed_tokens = process_text(text, stopwords, punctuations_special_char)
    return stemmed_tokens


'''The below Function utilizes Multiprocessing to read the documents in the document directory and calls
 the process_text_file function every time'''


def read_documents_using_multithreading(document_directory, stopwords, punctuation_special_character_file_path):
    files = [os.path.join(document_directory, filename) for filename in os.listdir(document_directory) if
             filename.endswith(".txt")]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        documents = list(
            executor.map(lambda file: process_text_file(file, stopwords, punctuation_special_character_file_path),
                         files))
    return documents


''' Function to build the inverted index term -> list of postings and it is reversed sorted such that
 it is there in a descending manner which helps to get the frequent terms in the beginning and then decreasing'''


def build_inverted_index(documents, file_list):
    inverted_index = defaultdict(lambda: {'collection_term_frequency': 0, 'document_frequency': 0, 'postings': set()})
    for doc_id, document in enumerate(documents):
        file_name = file_list[doc_id]
        # keeping a set of unique terms in the document just to keep track
        seen_terms = set()
        # Increment term Frequency and update document frequency
        for term in document:
            inverted_index[term]['collection_term_frequency'] += 1
            seen_terms.add(term)
        # update document frequency and add document name in the postings
        for term in seen_terms:
            inverted_index[term]['document_frequency'] += 1
            inverted_index[term]['postings'].add(file_name)
    sorted_inverted_index = {key: value for key, value in
                             sorted(inverted_index.items(), key=lambda items: len(items[1]['postings']), reverse=True)}
    return sorted_inverted_index


''' Function to get the size of the inverted index it is to be noted that while calculating the size it is small 
but while i am saving it into a file there are other metadatas that are being saved as well as the inverted index
also while saving the inverted index the document frequency is also included '''


def size_of_inverted_index(inverted_index):
    # Calculate the size of the inverted index in bytes
    size = inverted_index.__sizeof__()
    print(
        f"Inverted Index Size: {size / (1024 ** 2):.2f} MB/ {size / (1024 ** 3):f} GB ")


'''This function takes in the n number of frequent terms and displays the term and their corresponding postings
 of documents and also the frequent terms with respect to collection term frequency '''


def display_top_n_frequent_terms(inverted_index, n):
    print(f"Top {n} frequent terms in the inverted index with document frequency:")
    print("{:<20} {:<20}".format("TERM", "Document Frequency"))
    count = 0
    for term, doc_ids in inverted_index.items():
        print("{:<20} {:<20}".format(term, doc_ids['document_frequency']))
        count += 1
        if count == n:
            count = 0
            break
    print(f"Top {n} frequent terms in the inverted index with collection term frequency:")
    print("{:<20} {:<20}".format("TERM", "Collection Term Frequency"))
    sorted_index = sorted(inverted_index.items(), key=lambda x: x[1]['collection_term_frequency'], reverse=True)
    for term, doc_ids in sorted_index:
        print("{:<20} {:<20}".format(term, doc_ids['collection_term_frequency']))
        count += 1
        if count == n:
            break


'''This function saves the inverted index according to the file name it is passed on to and contains json structure 
containing the terms the document frequency and the the posting list i.e the document list '''


def save_inverted_index(index, filename):
    # Convert sets to lists before saving
    inverted_index_to_save = {}
    for term, info in index.items():
        inverted_index_to_save[term] = {
            'collection_term_frequency': info['collection_term_frequency'],
            'document_frequency': info['document_frequency'],
            'postings': list(info['postings'])
        }
    with open(filename, 'w') as file:
        json.dump(inverted_index_to_save, file)
    print("Inverted Index saved to file")


'''The function below loads the inverted index from a text file'''


def load_inverted_index(filename):
    with open(filename, 'r') as file:
        return json.load(file)


if __name__ == "__main__":
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
    documents = read_documents_using_multithreading(documents_path, stop_words, punctuations_special_char)
    # building the inverted index
    inverted_index = build_inverted_index(documents, files)
    # prints out the size of the inverted index in megabyte and gigabyte
    # (note: this is the size of the inverted index only not the inverted index with the document frequency )
    size_of_inverted_index(inverted_index)
    # prints the n number of top frequent items
    display_top_n_frequent_terms(inverted_index, 5)
    # saves the inverted index with the document frequency
    save_inverted_index(inverted_index, "./inverted_index.txt")
    # Loads the inverted index from a txt file
    loaded_inverted_index = load_inverted_index("inverted_index.txt")
