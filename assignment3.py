import re
import math
import random

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

# the unigram model will hold no prior context to the next token in the sequence.
# it is an ngram model where n is 1
def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.
    # Replace the line below with your code.
    vocabulary = {}
    for k in sequence:
        # when key is not in vocabulary, initialise key with value 1
        if k not in vocabulary:
            vocabulary[k] = 1
        # when key is present in vocabulary, increment key's value by 1
        else:
            vocabulary[k] = vocabulary[k] + 1
    # a unigram is an empty tuple with value that is the vocabulary dictionary
    unigram = {(): vocabulary}
    return unigram

# the bigram model will hold one token as context to the next token in the sequence.
# it is an ngram model where n is 2
def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.
    # Replace the line below with your code.
    bigram = {}
    for i,k in enumerate(sequence):
        # generating context with indexing should not exceed the length of the sequence 
        # the i specifies the current index being iterated
        # the +1 accounts for the next token
        if i+1 < len(sequence):
            tuple_key = tuple([k,''])
            next_token = sequence[i+1]
            # when the tuple key is not in the bigram, initialise the key
            # with its value set as a dictionary containing the key/value pair
            # of the next token with a count of 1
            if tuple_key not in bigram:
                bigram[tuple_key] = {sequence[i+1]: 1}
            # when the tuple key already exists in the bigram, then check for
            # whether the next token exists
            else:
                # when the next token does not exist, initialise the key/value
                # pair of the next token with a count of 1
                if next_token not in bigram[tuple_key]:
                    bigram[tuple_key][next_token] = 1
                # when the next token already exists in the dictionary, increment
                # its value by 1
                else:
                    bigram[tuple_key][next_token] = bigram[tuple_key][next_token] + 1
        # break when indexing would exceed the length of the sequence
        else:
            break
    return bigram

# the ngram model will hold n-1 number of tokens as context to the next token in the sequence
def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    # Replace the line below with your code.
    ngram = {}
    for i,k in enumerate(sequence):
        # generating context with indexing should not exceed the length of the sequence
        if i+n-1 < len(sequence):
            # specify the tuple key as being a sequence of n-1 tokens from the curent index i
            # the next token must be downstream from the context so i+n-1
            tuple_key = tuple([sequence[j] for j in range(i,i+n-1)])
            next_token = sequence[i+n-1]
            # when the tuple key does not exist in the ngram, initialise the key
            # with its value set as a dictionary containing the key/value pair
            # of the next token with a count of 1
            if tuple_key not in ngram:
                ngram[tuple_key] = {next_token: 1}
            # when the tuple key already exists in the ngram,
            # then check for whether the next token exists
            else:
                if next_token not in ngram[tuple_key]:
                    ngram[tuple_key][next_token] = 1
                else:
                    ngram[tuple_key][next_token] = ngram[tuple_key][next_token] + 1
        else:
            break
    return ngram

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    if () in model:
        return model[()]
    elif () not in model and len(model.keys()) >= 2:
        if tuple(sequence) in model.keys():
            return model[tuple(sequence)]
        else:
            return None
    else:
        raise Exception("Have you passed an ngram model and a sequence as arguments?")

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor
    comp = 1 - factor
    for pred in preds[:-1]:
        if pred:
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            mult = comp * factor
            comp -= mult
    pred = preds[-1]
    mult += comp
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    # '''
    model = build_unigram(sequence[:20])
    print(model)
    # '''

    # Task 1.2 test code
    # '''
    model = build_bigram(sequence[:20])
    print(model)
    # '''

    # Task 1.3 test code
    # '''
    model = build_n_gram(sequence[:20], 5)
    print(model)
    # '''

    # Task 2 test code
    # '''
    print(query_n_gram(model, tuple(sequence[:4])))
    # '''

    # Task 3 test code
    '''
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    '''

    # Task 4.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 4.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''

