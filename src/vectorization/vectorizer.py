from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import vectorization.features_vectorizer as features_vectorizer
import vectorization.embeddings_vectorizer  as embeddings_vectorizer
import util.const as const

class Vectorizer():
    '''
    Vectorizer interface
    '''

    # Constructor
    def __init__(self, vectorization=const.VECTORIZERS['one_hot']):

        # Type of vectorizer
        self.type = vectorization

        # Empty objects
        self.vectorizer = None
        self.dictionary = None

        # Depending on vectorization type, initializes different objects
        if self.type == const.VECTORIZERS['one_hot']:
            self.vectorizer = CountVectorizer()
        if self.type == const.VECTORIZERS['features']:
            self.vectorizer = DictVectorizer()
        if self.type == const.VECTORIZERS['word_embeddings']:
            self.dictionary = embeddings_vectorizer.get_dictionary()

    # Given a set X, vectorizes it and saves used vectorizer
    def fit(self, X):

        vectorized = []

        # If vectorization is one hot encoding, uses CountVectorizer
        if self.type == const.VECTORIZERS['one_hot']:
            vectorized = self.vectorizer.fit_transform(X).toarray()

        # If vectorization is by features, uses feature extraction and DictVectorizer
        if self.type == const.VECTORIZERS['features']:
            featurized = features_vectorizer.get_features(X)
            vectorized = self.vectorizer.fit_transform(featurized)

        # If vectorization is by word embeddings, uses word embeddings dictionary and mean
        if self.type == const.VECTORIZERS['word_embeddings']:
            vectorized = embeddings_vectorizer.get_vectors(X, self.dictionary)

        return vectorized

    # Given a set X, vectorizes it using last vectorizer
    def transform(self, X):
        
        vectorized = []

        # If vectorization is one hot encoding, uses CountVectorizer        
        if self.type == const.VECTORIZERS['one_hot']:
            vectorized = self.vectorizer.transform(X).toarray()

        # If vectorization is by features, uses feature extraction and DictVectorizer        
        if self.type == const.VECTORIZERS['features']:
            featurized = features_vectorizer.get_features(X)
            vectorized = self.vectorizer.transform(featurized)

        # If vectorization is by word embeddings, uses word embeddings dictionary and mean        
        if self.type == const.VECTORIZERS['word_embeddings']:
            vectorized = embeddings_vectorizer.get_vectors(X, self.dictionary)

        return vectorized