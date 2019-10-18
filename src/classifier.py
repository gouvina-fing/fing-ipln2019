import os
import sys
from model import Model
import util.features as features

# MAIN FUNCTIONS

# Given a list of examples, predict its classification using default model
def predict(examples):

    # 1. Create model
    model = Model()
    print('(CLASSIFIER) Creating model...')

    # 2. Load classifier
    model.load()
    print('(CLASSIFIER) Loading model...')

    # 3. Make prediction
    prediction = model.predict(examples)
    print('(CLASSIFIER) Prediction obtained (' + str(prediction) + ')')

    return prediction

if __name__ == "__main__":

    # 1. Get text from args
    text = sys.argv[1]
    texts = [text]

    # 2. Predict category
    main_prediction = predict(texts)
