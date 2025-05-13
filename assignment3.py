import re
import math
import random
import numpy as np

def tokenise(filename):
    # include encoding as `utf-8-sig`` otherwise tokenise method finds /ufeff as a first token
    with open(filename, 'r', encoding='utf-8-sig') as f:
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

# Given a query, ask the ngram model to return a prediction.
# The prediction is a dictionary. 
# If the context is too small for the ngram model, the model returns None (no prediction).
def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    
    # unigram case
    if () in model:
        return model[()]
    # n_gram case, bigram and up:
    elif () not in model and len(model.keys()) >= 2:
        if tuple(sequence) in model.keys():
            return model[tuple(sequence)]
        else:
            return None
    else:
        raise Exception("Have you passed an ngram model and a sequence as arguments?")

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor # first iteration 0.8
    comp = 1 - factor # first iteration 0.2
    # for all predictions except the last element
    for pred in preds[:-1]:
        # Only valid predictions are considered, None predictions are discarded.
        # Predictions are a dictionary with keys as words and their frequency of
        # appearing as values in relation to their context
        if pred:
            # for the given prediction there will be a dictionary, with a possible
            # n keys with natural numbers as their value; sum all the values up
            # to calculate a weight_sum
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                # if predicted value already exists in blended probabilities
                # then add to it
                # else the predicted value does not exist in the blended probabilities
                # then initialise it
                # in this given prediction, the summation of all v / weight_sum will equal 1
                # therefore the total probability
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            # second iteration mult would be 0.8 * 0.2 -> 0.16
            # second iteration comp would be 0.2 - 0.16 -> 0.04
            mult = comp * factor
            comp -= mult
    # handling for last prediction since probabilities must add up to one
    # for example when there are only two predictions
    # then the first factor will be weighted 0.8
    # and the second factor will be weighted 0.2
    pred = preds[-1]
    mult += comp # 0.16 + 0.04 -> 0.2
    # then carry out the same operation as above to get the blended probabilities
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    # weighted sum will be equal to n dictionaries processed
    # calculate weight sum again to rebalance to a probability summing to 1
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    
    # query all models to return a prediction for the given sequence
    # preds = [query_n_gram(model,sequence) for model in models]
    preds = []
    for model in models:
        min_serviceable_tokens = len(next(iter(model)))
        if len(sequence) < min_serviceable_tokens:
            preds.append(None)
        else:
            preds.append(query_n_gram(model, sequence[-min_serviceable_tokens:]))

    blended_probs = blended_probabilities(preds)
    choice = np.random.choice(list(blended_probs.keys()),1,replace=False,p=list(blended_probs.values()))
    token = str(choice[0])
    return token

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    log_likelihood_list = []
    for i in range(0,len(sequence)):
        # ramp up from unigram model up to 10-gram model
        # tuple_key holds up to i tokens as context
        if i < len(models):
            model = list(reversed(models))[i]
            tuple_key = tuple(sequence[:i])
        # keep using the 10-gram model once reached
        # tuple_key holds the last 9 tokens as context
        else:
            model = list(reversed(models))[len(models)-1]
            tuple_key = tuple(sequence[i-len(models)+1:i])

        # find the frequency that the next_token appears
        # in the given n-gram for the given key
        next_token = sequence[:i+1][-1]
        try:
            frequency = model[tuple_key][next_token]
        except KeyError:
            return -math.inf

        # find the frequency of its appearance relative
        # to the frequency of all other tokens appearing
        # in the given n-gram for the given key
        weight_sum = sum(model[tuple_key].values())
        likelihood = frequency / weight_sum
        log_likelihood = math.log(likelihood)
        log_likelihood_list.append(log_likelihood)

    return sum(log_likelihood_list)


def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    log_likehood_list = []
    
    # iteration over the sentence
    for i in range(0,len(sequence)):
        # next token to predict
        next_token = sequence[:i+1][-1]
        
        if i < len(models):
            model_limit = i + 1
        else:
            model_limit = len(models)

        active_predictions = []

        # iteration over all legal n-grams
        for j in range(0,model_limit):
            model = list(reversed(models))[j]
            # when just the one-gram, set tuple key to an empty tuple
            if () in model: 
                tuple_key = tuple([])
            # all other n-grams will have a tuple key based on the
            # moving read across the sequence of tokens
            # slice the sequence to END at the latest read :i
            # slice the sequence to START at the possible of context
            # that the jth n-gram can hold
            # i-j: should not experience an IndexOutOfBounds exception since
            # j is constrained by the model limit.
            else:
                tuple_key = tuple(sequence[i-j:i]) 

            try:
                active_predictions.append(model[tuple_key])
            except KeyError:
                return -math.inf

        # all the available and legal n_gram predictions with the
        # highest order n-gram at the zero index are used
        # i.e. a dictionary containing n dictionary of predictions
        active_predictions = list(reversed(active_predictions))
        blended_probs = blended_probabilities(active_predictions)
        likelihood = blended_probs[next_token]
        log_likelihood = math.log(likelihood)
        log_likehood_list.append(log_likelihood)

        # reset active_predictions for next iteration
        active_predictions = []
    
    # log of products is the same as the log of each term added together
    return sum(log_likehood_list)

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    # '''
    print("Unigram Model:")
    model = build_unigram(sequence[:20])
    print(model)
    print()
    # '''

    # Task 1.2 test code
    # '''
    print("Bigram Model:")
    model = build_bigram(sequence[:20])
    print(model)
    print()
    # '''

    # Task 1.3 test code
    # '''
    print("N gram model:")
    model = build_n_gram(sequence[:20], 5)
    print(model)
    print()
    # '''

    # Task 2 test code
    # '''
    print("Query N gram model:")
    print(query_n_gram(model, tuple(sequence[:4])))
    print()
    # '''

    # Task 3 test code
    # '''
    # Build ngram in reverse order since more context should result in better predictions
    # And when blending probabilities together, the more complex ngram has higher weighting.
    # Head will initially be empty because no predictions have been made.
    # But as predictions are outputted they are appended to head and used as the next input.
    print("Sample blended N gram models:")
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    print()
    # '''

    # Task 4.1 test code
    # '''
    print("Log likelihood ramp up:")
    print(log_likelihood_ramp_up(sequence[:20], models))
    print()
    # '''

    # Task 4.2 test code
    # '''
    print("Log likelihood from blended models:")
    print(log_likelihood_blended(sequence[:20], models))
    # '''

