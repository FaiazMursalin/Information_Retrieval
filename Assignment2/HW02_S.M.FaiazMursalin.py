# if you don't have nltk setup then for the first time while you are setting up you need to do a nltk.download('punkt')
import nltk as tk
# Importing the necessary functions from the modified HW01 code
from modified_HW01_S_M_FaiazMursalin import load_inverted_index, read_stop_words, read_punctuations_special_char
import concurrent.futures as cf  # imported for using multithreading features for reading documents in the directory
import itertools

'''Function to display course and student information'''


def display_information():
    print("=================== CSC790-IR Homework 02 ==============")
    print("First Name: S.M. Faiaz")
    print("Last Name: Mursalin")
    print("========================================================")


''' Function to load user queries from a file into a list '''


def load_user_queries(filename):
    with open(filename, "r") as file:
        content = file.read()
    return content.strip().split('\n')


''' Function which does the following:
    a. reads a query
    b. tokenizes the entire query
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


''' Generate all possible combinations of query terms and operators'''


def combinations(result):
    # Define operators
    operators = ['AND', 'OR']
    # Get query terms
    terms = list(result.keys())
    # Calculate the number of operators needed
    number_of_operators = len(terms) - 1
    # Initialize list to store combinations
    combinations_list = []
    # Generate all possible combinations of operators for the query terms
    for operator_combination in itertools.product(operators, repeat=number_of_operators):
        combination = []
        combination.append(terms[0])
        for i, operator in enumerate(operator_combination):
            combination.append(operator)
            combination.append(terms[i + 1])
        combinations_list.append(combination)

    # Initialize dictionary to store combinations
    combinations_dict = {}
    # Evaluate each combination and store the results
    for combination in combinations_list:
        postings = evaluate_combination(combination, result.copy())
        combinations_dict[tuple(combination)] = postings
    return combinations_dict


''' Evaluate a single combination of query terms and operators'''


def evaluate_combination(combination, postings):
    # Initialize result with postings of the first term
    result = postings[combination[0]]
    # Iterate through the combination, applying operators to the postings of subsequent terms
    for i in range(1, len(combination), 2):
        operator = combination[i]
        term = combination[i + 1]
        if operator == "AND":
            result = set(result) & set(postings[term])
        elif operator == "OR":
            result = set(result) | set(postings[term])
        else:
            raise ValueError(f"Invalid operator: {operator}")

    return list(result)


'''Search the inverted index for a given query'''


def search_query(query, inverted_index, stopwords, punctuations_special_char):
    # Process the query text
    processed_query = process_text(query, stopwords, punctuations_special_char)
    # Initialize dictionary to store query results
    result = dict()
    # Iterate through each processed term in the query
    for term in processed_query:
        # Retrieve the postings list for the term from the inverted index, or an empty list if not found
        if term in inverted_index:
            result[term] = inverted_index[term]['postings']
        else:
            result[term] = []

    # Generated combinations of query terms and operators
    search_result = combinations(result)
    return search_result


'''Print the search results for a given query'''


def print_search_results(query_number, query, results):
    print(f"================== User Query {query_number}: {query} =================")
    for result in results:
        print(f"============= Results for: {result} =================")
        print(", ".join(results[result]))


if __name__ == '__main__':
    # display course and students information of myself and what course
    display_information()
    # load the saved inverted index
    inverted_index = load_inverted_index('./inverted_index.txt')
    # relative file path for user queries
    query_path = './special_chars-queries/queries.txt'
    # load the user queries from a file
    queries = load_user_queries(query_path)
    # Read stop words and punctuation/special characters
    stopwords = read_stop_words("./stopwords.txt")
    punctuations_special_char = read_punctuations_special_char("./special_chars-queries/special-chars.txt")
    # process queries in parallel
    with cf.ThreadPoolExecutor() as executor:
        search_results = executor.map(
            lambda query: search_query(query, inverted_index, stopwords, punctuations_special_char),
            queries)
    # Print the search results for each query
    for i, search_result in enumerate(search_results):
        print_search_results(i+1, queries[i], search_result)
