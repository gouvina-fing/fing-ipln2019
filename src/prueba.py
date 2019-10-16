import util.processWordEmbeddings as pwe


if __name__ == "__main__":

    tweet = '¡Ganó Colón! Con gol de Javier Correa, venció 1-0 como local a Gimnasia (expulsado Maximiliano Coronel) en la fecha 16 de la Superliga'
    print('cargando embeddings')
    emb = pwe.load_embeddings()
    print('embeddings cargado')

    print(emb.get('0'))

    print('Tweet: ' + tweet)

    res = pwe.convert_tweet_to_embedding(tweet, emb)
    #import pdb; pdb.set_trace()
    print(res)