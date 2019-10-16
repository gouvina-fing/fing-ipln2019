import sys
import pandas as pd
import numpy as np
import statistics
import util.const as const
from tokenizer import tokenize

def load_embeddings():
    dicc = {}

    # Process embeddings file
    # Read file as Pandas DataFrame
    df_test = pd.read_csv(const.DATA_FOLDER + const.EMBEDDINGS_FILE, engine='python', sep='\s+', header=None) 
    
    # Get words and embeddings values
    
    rows = df_test.shape[0] -1
    words = df_test.loc[0:rows, 0]


    df = df_test.loc[0:rows, 1:300]

    df.fillna(0)

    for index, row in df.iterrows():
        if not(words[index] in dicc.keys()):
            dicc[words[index]] = row.values
    return dicc

def convert_tweet_to_embedding(tweet, embeddings):

    words = np.array(tokenize_text(tweet), dtype=object)
    return mean_of_tweet_embedding(words, embeddings)

def tokenize_text(text):

    # Eliminate symbols
    symbols = ['.',',','_',';','"','\n',"'",'!',':','?']
    for symbol in symbols:
        text = text.replace(symbol, ' ')

    # Tokenize
    words = [ token.txt for token in tokenize(text) if  token.txt is not None]

    return words


def mean_of_tweet_embedding(array_of_words, embeddings):
    for i, elem in enumerate(array_of_words):
        array_of_words[i] = token_to_embedding(elem, embeddings)
    each_index_array = list(zip(*array_of_words))
    for i, elem in enumerate (each_index_array):
        each_index_array[i] = statistics.mean(elem)
    return each_index_array
    '''
    try:
        array_of_arrays = np.vectorize(token_to_embedding)(array_of_words, embeddings)
    except:
        import pdb; pdb.set_trace()

    each_index_array = list(zip(*array_of_arrays))
    return np.vectorize(statistics.mean(each_index_array))

    '''
    '''
    result = np.zeros(300)
    for word in array_of_words:
        if word in embeddings.keys():
            result += embeddings.get(word)
    if len(array_of_words) != 0:
        result = result / len(array_of_words)
    return result
    '''

def token_to_embedding(word, embeddings):
    if word in embeddings.keys():
        return embeddings.get(word)
    else:
        return np.zeros(300)
    