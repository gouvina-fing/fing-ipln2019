import sys
import pandas as pd
import numpy as np
import statistics
import util.const as const
import re
from tokenizer import tokenize


# MAIN FUNCTIONS


#returns the csv embeddings
def get_dictionary():
    return load_embeddings()

# returns the mean embeddign for each tweet from the set
def get_vectors(tweets, dictionary):

    res = np.array([np.zeros(300)])
    '''
    SI USO ESTE METODO EXPLOTA EN EL FIT

    self.dataset =  np.array(self.dataset, dtype=object)
    for i, tweet in enumerate(self.dataset):
        try: 
            self.dataset[i] = np.array(pWorEmb.convert_tweet_to_embedding(tweet, self.embeddings), dtype=object)
        except:
            import pdb; pdb.set_trace()
            print('errror')
    
    '''    
    for tweet in tweets:
        e = convert_tweet_to_embedding(tweet, dictionary)
        res = np.concatenate((res,[e]))
    res = res[1:]
    tweets = res
    
    #tweets = np.vectorize(convert_tweet_to_embedding)(tweets,dictionary)
    return tweets

# AUXILIAR FUNCTIONS

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

    # Eliminate symbols --- SE PUEDE USAR UNA ER
    regex = r"¡|!|,|\?|\.|=|\+|-|_|&|\^|%|$|#|@|\(|\)|`|'|<|>|/|:|;|\*|$|¿|\[|\]|\{|\}|~"
    text = re.sub(regex, ' ', text)


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
    

def token_to_embedding(word, embeddings):
    if word in embeddings.keys():
        return embeddings.get(word)
    else:
        return np.zeros(300)
