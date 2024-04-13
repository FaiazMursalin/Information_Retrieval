# if you don't have nltk setup then for the first time while you are setting up you need to do a nltk.download('punkt')
import nltk as tk
import os
import concurrent  # imported for using multithreading features for reading documents in the directory
from collections import defaultdict  # imported for using default dictionary
import json  # imported for saving and loading the inverted index

'''Function to display course and student information'''


def display_information():
    print("=================== CSC790-IR Homework 01 ==============")
    print("First Name: S.M. Faiaz")
    print("Last Name: Mursalin")
    print("========================================================")


''' Function to read the stop words from the stopword.txt file and tokenize them and return the list of stop words'''


def read_stop_words(stop_words_file):
    with open(stop_words_file, "r") as file:
        content = file.read()
        stop_word_tokens = tk.word_tokenize(content)
    return stop_word_tokens


''' Function which reads a document and tokenizes the entire document, perform stemming and later remove stop words'''


def process_text_file(filepath, stopwords):
    stemmer = tk.stem.PorterStemmer()
    with open(filepath, 'r') as file:
        text = file.read()
        tokens = tk.word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        tokens = [stemmed_token for stemmed_token in stemmed_tokens if stemmed_token not in stopwords]
    return tokens


'''The below Function utilizes Multiprocessing to read the documents in the document directory and calls
 the process_text_file function every time'''


def read_documents_using_multithreading(document_directory, stopwords):
    files = [os.path.join(document_directory, filename) for filename in os.listdir(document_directory) if
             filename.endswith(".txt")]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        documents = list(executor.map(lambda file: process_text_file(file, stopwords), files))
    return documents


''' Function to build the inverted index term -> list of postings and it is reversed sorted such that
 it is there in a descending manner which helps to get the frequent terms in the beginning and then decreasing'''


def build_inverted_index(documents, file_list):
    inverted_index = defaultdict(set)

    for doc_id, document in enumerate(documents):
        file_name = file_list[doc_id]
        for term in document:
            inverted_index[term].add(file_name)
    sorted_inverted_index = {key: value for key, value in
                             sorted(inverted_index.items(), key=lambda items: len(items[1]), reverse=True)}
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
 of documents '''


def display_top_n_frequent_terms(n):
    print(f"Top {n} frequent terms in the inverted index:")
    count = 0
    for term, doc_ids in inverted_index.items():
        print(f"{term}: {list(doc_ids)}")
        count += 1
        if count == n:
            break


'''This function saves the inverted index according to the file name it is passed on to and contains json structure 
containing the terms the document frequency and the the posting list i.e the document list '''


def save_inverted_index(index, filename):
    # Convert sets to lists before saving
    # index = {term: list(doc_ids) for term, doc_ids in index.items()}
    index = {term: {'doc_frequency': len(doc_ids), 'doc_ids': list(doc_ids)} for term, doc_ids in index.items()}
    with open(filename, 'w') as file:
        json.dump(index, file)
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
    # Read the documents using multithreading
    documents = read_documents_using_multithreading(documents_path, stop_words)
    # building the inverted index
    inverted_index = build_inverted_index(documents, files)
    # prints out the size of the inverted index in megabyte and gigabyte
    # (note: this is the size of the inverted index only not the inverted index with the document frequency )
    size_of_inverted_index(inverted_index)
    # prints the n number of top frequent items
    display_top_n_frequent_terms(5)
    # saves the inverted index with the document frequency
    save_inverted_index(inverted_index, "./inverted_index.txt")
    # Loads the inverted index from a txt file
    loaded_inverted_index = load_inverted_index("inverted_index.txt")
