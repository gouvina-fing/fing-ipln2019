import sys
import pandas as pd
import numpy as np
import util.const as const

if __name__ == "__main__":
    dicc = {}

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
    # 3.1 Split text by ''
    words = text.split()
    print(words)
    # 3.2 Search if a symbol is together to a word.

    # 4. get de average of all the embeddings of the sentence
    result = np.zeros(300)
    for word in words:
        #CHEQUEAR QUE LA PALABRA ESTE EN EL DICCIONARIO
        result += dicc[word]
    print(result)

