import os
import sys
import util.const as const
from model import Model

# MAIN FUNCTIONS

# Train a model
def train():

    # 1. Create model
    model = Model()
    print('(TRAINER) Model created')

    # 2. Train classifier
    model.train()
    print('(TRAINER) Model trained')

    # 3. Save classifier
    model.save()
    print('(TRAINER) Model saved')

    return model

if __name__ == "__main__":
    train()
