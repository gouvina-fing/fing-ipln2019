import os

MAIN_ROUTE = str(os.getcwd()).replace('/src', '')

# FOLDERS AND FILES
# ---------------------------------------------------------------

DATA_FOLDER = MAIN_ROUTE + "/data"
DATA_TRAIN_FILE = "/data_train.csv"
DATA_TEST_FILE = "/data_test.csv"
DATA_VAL_FILE = "/data_val.csv"

MODEL_FOLDER = MAIN_ROUTE + "/models"
MODEL_FILE = "/model.sav"

# EVALUATOR
# ---------------------------------------------------------------

EVALUATIONS = {
    'none': 0,
    'normal': 1,
    'cross': 2,
}

# MODEL
# ---------------------------------------------------------------

MODELS = ['svm', 'tree', 'nb', 'knn', 'mlp_classifier']
