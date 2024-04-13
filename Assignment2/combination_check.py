from itertools import product

def combinations(result):
    """
    Generates all possible combinations of logical operators (AND, OR) between the keys of a dictionary.

    Args:
        result (dict): A dictionary where keys are terms and values are postings.

    Returns:
        list: A list of lists, where each inner list represents a combination of operators and terms.
    """

    operators = ['AND', 'OR']
    terms = list(result.keys())
    num_operators = len(terms) - 1

    combinations_list = []
    for operator_combination in product(operators, repeat=num_operators):
        combination = []
        combination.append(terms[0])
        for i, operator in enumerate(operator_combination):
            combination.append(operator)
            combination.append(terms[i+1])
        combinations_list.append(combination)

    return combinations_list

def evaluate_combination(combination, postings):
    """
    Evaluates a combination of operators and terms on a dictionary of postings.

    Args:
        combination (list): A list of terms and operators.
        postings (dict): A dictionary where keys are terms and values are postings.

    Returns:
        list: The resulting postings after applying the operators.
    """

    result = postings[combination[0]]
    for i in range(1, len(combination), 2):
        operator = combination[i]
        term = combination[i+1]
        if operator == "AND":
            result = set(result) & set(postings[term])
        elif operator == "OR":
            result = set(result) | set(postings[term])
        else:
            raise ValueError(f"Invalid operator: {operator}")

    return list(result)

# Example usage
result = {'apple': [1, 2, 3], 'banana': [4, 5], 'orange': [6, 7]}
combinations_list = combinations(result)
print(combinations_list)

for combination in combinations_list:
    postings = evaluate_combination(combination, result.copy())
    print(f"Combination: {combination}, Postings: {postings}")

