import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import util.const as const
import util.processWordEmbeddings as pWorEmb

class Model():
    '''
    Model representation
    '''

    # Create dataset in csv format
    def read_dataset(self):

        # Read dataset as Pandas DataFrame
        df = pd.read_csv(const.DATA_FOLDER + const.DATA_TRAIN_FILE)

        # Shuffle dataset before spliting columns
        df = df.sample(frac=1)

        # Get train dataset and train categories
        self.dataset = df['text'].values.astype('U')
        self.categories = df['humor'].values

        # If there is evaluation, get train dataset too
        if self.evaluation != const.EVALUATIONS['none']:
            
            # Read dataset as Pandas DataFrame
            df_test = pd.read_csv(const.DATA_FOLDER + const.DATA_TEST_FILE)

            # Shuffle dataset before spliting columns
            df_test = df_test.sample(frac=1)

            # Get train dataset and train categories
            self.test_dataset = df_test['text'].values.astype('U')
            self.test_categories = df_test['humor'].values

    # Tokenize texts for input to model
    def tokenize_dataset(self):
        self.vectorizer = CountVectorizer()
        self.dataset = self.vectorizer.fit_transform(self.dataset).toarray()
    
    # Load embeddings.csv
    def load_embeddings(self):
        self.embeddings = pWorEmb.load_embeddings()

    # Transform the dataset of tweets into its mean vector of embeddings asociated 
    def dataset_embedded(self):
        self.load_embeddings()
        res = np.array([np.zeros(300)])
        '''
        self.dataset =  np.array(self.dataset, dtype=object)
        for i, tweet in enumerate(self.dataset):
            try: 
                self.dataset[i] = np.array(pWorEmb.convert_tweet_to_embedding(tweet, self.embeddings), dtype=object)
            except:
                import pdb; pdb.set_trace()
                print('errror')
        print('FIN 1')

        import pdb; pdb.set_trace()
        print('FIN')
        '''    
        for tweet in self.dataset:
            e = pWorEmb.convert_tweet_to_embedding(tweet, self.embeddings)
            res = np.concatenate((res,[e]))
        res = res[1:]
        self.dataset = res
        
        #self.dataset = np.vectorize(pWorEmb.convert_tweet_to_embedding)(self.dataset,self.embeddings)

        '''
        '''
        '''
        try:
            # self.dataset = np.vectorize(pWorEmb.convert_tweet_to_embedding)(self.dataset,self.embeddings)
            res = np.vectorize(pWorEmb.convert_tweet_to_embedding)(self.dataset,self.embeddings)
        except BaseException as  error:
            print('hubo un error')
            # import pdb; pdb.set_trace()

            print(str(error))

        '''
        # Vectorize texts for input to model
    def vectorize_dataset(self):
        self.vectorizer = Vectorizer(self.vectorization)
        self.dataset = self.vectorizer.fit(self.dataset)

    # Aux function - For saving classifier
    def save(self):
        pickle.dump(self.classifier, open(const.MODEL_FOLDER + const.MODEL_FILE, 'wb'))

    # Aux function - For loading classifier
    def load(self):
        self.classifier = pickle.load(open(const.MODEL_FOLDER + const.MODEL_FILE, 'rb'))

    # Constructor
    def __init__(self, model='mlp_classifier', evaluation=const.EVALUATIONS['none']):

        # Create empty dataset for training
        self.dataset = None
        self.categories = None

        # Create empty testset for evaluation
        self.test_dataset = None
        self.test_categories = None

        # Create other empty objects
        self.classifier = None
        self.vectorizer = None
        self.model = model

        # Generate default values
        self.threshold = 0.5
        self.evaluation = evaluation
        self.evaluation_normal_size = 0.2
        self.evaluation_cross_k = StratifiedKFold(10, True)

        # Read dataset and categories
        self.read_dataset()

        # Tokenize dataset and save vectorizer
        self.dataset_embedded()

    # Create and train classifier depending on chosen model
    def train(self):
        if self.model == 'svm':
            self.classifier = SVC(gamma='auto', probability=True)
        if self.model == 'tree':
            self.classifier = DecisionTreeClassifier(max_depth=5)
        if self.model == 'nb':
            self.classifier = GaussianNB()
        if self.model == 'knn':
            self.classifier = KNeighborsClassifier(5)
        elif self.model == 'mlp_classifier':
            self.classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, solver='sgd')

        # Train using dataset
        self.classifier.fit(self.dataset, self.categories)

    # Predict classification for X using classifier
    def predict(self, X):

        # Vectorize text
        examples = self.vectorizer.transform(X)

        # Generate classification and probabilities for every class
        prediction = self.classifier.predict(examples)

        return prediction

    # Generate evaluation depending of type
    def evaluate(self):

        if self.evaluation == const.EVALUATIONS['normal']:
            return self.normal_evaluate()

        elif self.evaluation == const.EVALUATIONS['cross']:
            return self.cross_evaluate()

        else:
            print('ERROR - (MODEL) Test dataset not generated')
            return None

    # Generate normal evaluation
    def normal_evaluate(self):

        prediction = self.predict(self.test_dataset)

        accuracy = accuracy_score(self.test_categories, prediction)
        results = classification_report(self.test_categories, prediction, output_dict=True)
        report_string = classification_report(self.test_categories, prediction)
        matrix = confusion_matrix(self.test_categories, prediction)

        report = {
            'f1_score': results['macro avg']['f1-score'],
            'precision': results['macro avg']['precision'],
            'recall': results['macro avg']['recall'],
        }

        return accuracy, report, report_string, matrix

    # Generate cross evaluation
    def cross_evaluate(self):

        results = cross_validate(self.classifier, self.dataset, self.categories,
                                cv=self.evaluation_cross_k, return_train_score=False,
                                scoring=('f1_micro', 'precision_micro', 'recall_micro', 'accuracy'))

        f1_score_list = results['test_f1_micro']
        precision_list = results['test_precision_micro']
        recall_list = results['test_recall_micro']
        accuracy_list = results['test_accuracy']

        report = {
            'f1_score': sum(f1_score_list) / len(f1_score_list),
            'precision': sum(precision_list) / len(precision_list),
            'recall': sum(recall_list) / len(recall_list),
        }
        accuracy = sum(accuracy_list) / len(accuracy_list)

        return accuracy, report, None, None
