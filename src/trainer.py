import os
import sys
import util.const as const
from model import Model

# MAIN FUNCTIONS

# Train a model
def train():

    tic = time.time()

    # 1. Create model
    model = Model()
    print('(TRAINER) Creating model...')


    # 2. Train classifier
    model.train()
    print('(TRAINER) Training model...')

    # 3. Save classifier
    model.save()
    print('(TRAINER) Saving model...')

    return model

if __name__ == "__main__":
    train()
