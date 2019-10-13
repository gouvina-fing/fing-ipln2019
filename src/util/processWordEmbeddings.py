import sys
import pandas as pd
import numpy as np
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

    for index, row in df.iterrows():
        dicc[words[index]] = row.values

    return dicc

def convert_tweet_to_embedding(tweet, embeddings):

    words = tokenize_text(tweet)
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
    result = np.zeros(300)
    for word in array_of_words:
        if word in embeddings:
            result += embeddings[word]
    if len(array_of_words) != 0:
        result = result / len(array_of_words)
    return result