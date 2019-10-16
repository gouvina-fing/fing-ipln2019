import os

MAIN_ROUTE = str(os.getcwd()).replace('/src', '')

# FOLDERS AND FILES
# ---------------------------------------------------------------

DATA_FOLDER = MAIN_ROUTE + "/data"
DATA_TRAIN_FILE = "/data_train.csv"
DATA_TEST_FILE = "/data_test.csv"
DATA_VAL_FILE = "/data_val.csv"
EMBEDDINGS_FILE = '/embeddings.csv'

MODEL_FOLDER = MAIN_ROUTE + "/models"
MODEL_FILE = "/model.sav"

# TYPES
# ---------------------------------------------------------------

MODELS = ['svm', 'tree', 'nb', 'knn', 'mlp_classifier']

VECTORIZERS = {
    'one_hot': 0,
    'features': 1,
    'word_embeddings': 2,
}

EVALUATIONS = {
    'none': 0,
    'normal': 1,
    'cross': 2,
}
