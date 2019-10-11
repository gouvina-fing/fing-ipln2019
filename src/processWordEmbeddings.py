import sys
import pandas as pd
import numpy as np
import util.const as const
from tokenizer import tokenize

if __name__ == "__main__":
    dicc = {}

    text = sys.argv[1]

    # 1. Process embeddings file
    # Read file as Pandas DataFrame
    df_test = pd.read_csv(const.DATA_FOLDER + '/embeddings.csv', engine='python', sep='\s+', header=None) 
    
    # Get words and embeddings values
    
    rows = df_test.shape[0] -1
    words = df_test.loc[0:rows, 0]


    df = df_test.loc[0:rows, 1:300]

    for index, row in df.iterrows():
        dicc[words[index]] = row.values # row values tiene que tomar el resto como un vector
    # 2. Get text from args
    text = sys.argv[1]

    # 3. Tokenize text
    # 3.1 eliminate symbols
    symbols = ['.',',','_',';','"','\n',"'",'!',]
    for symbol in symbols:
        text = text.replace(symbol, ' ')

    # 3.2 tokenize
    words = [ token.txt for token in tokenize(text) if  token.txt is not None]
    print(words)

    # 4. get de average of all the embeddings of the sentence
    result = np.zeros(300)
    for word in words:
        if word in dicc:
            result += dicc[word]
    print('suma: ')
    print(result)
    result = result / len(words)
    print('promedio: ')
    print(result)
