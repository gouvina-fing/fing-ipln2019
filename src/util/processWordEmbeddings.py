import sys
import pandas as pd
import numpy as np
import statistics
import re
import const as const
from tokenizer import tokenize


def load_embeddings():
    dicc = {}

    # Process embeddings file
    # Read file as Pandas DataFrame
    df_test = pd.read_csv('/home/gonzalo/Documents/IPLN/ipln2019/data/embeddings.csv', engine='python', sep='\s+', header=None) 
    
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

    tic = time.time()

    words = np.array(tokenize_text(tweet), dtype=object)
    a = mean_of_tweet_embedding(words, embeddings)

    toc = time.time()
    print('tiempo ' + str(toc - tic))
    return a
def tokenize_text(text):

    # Eliminate symbols --- SE PUEDE USAR UNA ER
    symbols = ['.',',','_',';','"','\n',"'",'!',':','?']
    for symbol in symbols:
        text = text.replace(symbol, ' ')
    regex = r"¡|!|,|\?|\.|=|\+|-|_|&|\^|%|$|#|@|\(|\)|`|'|<|>|/|:|;|\*|$|¿|\[|\]|\{|\}|~"
    text = re.sub(regex, ' ', text)


    # Tokenize
    words = [ token.txt for token in tokenize(text) if  token.txt is not None]
    return words


def mean_of_tweet_embedding(array_of_words, embeddings):

'''
    for i, elem in enumerate(array_of_words):
        a[i] = token_to_embedding(elem, embeddings)

    each_index_array = list(zip(*a))

    for i, elem in enumerate (each_index_array):
        each_index_array[i] = statistics.mean(elem)
    '''
    
    try:

        data = pd.Series(array_of_words)
        data = data.apply(token_to_embedding,embeddings=embeddings)
        each_index_array = list(zip(*data))
        each_index_array = list(map(statistics.mean,each_index_array))
        each_index_array = np.array(each_index_array)

    except:
        import pdb; pdb.set_trace()
    
    return each_index_array
    

def token_to_embedding(word, embeddings):
    if word in embeddings.keys():
        return embeddings.get(word)
    else:
        return np.zeros(300)
