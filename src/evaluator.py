# DEPENDENCIES
import sys
import util.const as const
from model import Model

# MAIN FUNCTIONS

# Train a model and evaluate its performance
def evaluate(hard_evaluation=False, evaluation=const.EVALUATIONS['normal'], grid_search=False):

    # Hard evaluation: Sort models by results and train the one with better results
    if hard_evaluation:

        # List of 5-uples (model, accuracy, precision, recall, f1_score)
        results_list = []

        for model_type in const.MODELS:

            for vectorizer in const.VECTORIZERS:
                
                print()
                print('(EVALUATOR) Evaluating model ' + model_type)
                print('(EVALUATOR) Evaluating vectorizer ' + vectorizer)

                # 1. Create model
                model = Model(vectorization=const.VECTORIZERS[vectorizer], model=model_type, evaluation=evaluation)

                # 2. Train classifier
                model.train(grid_search=grid_search)

                # 3. Evaluate classifier
                accuracy, results, _, _ = model.evaluate()
                results_list.append((vectorizer, model_type, accuracy, results['precision'], results['recall'], results['f1_score']))

        # Sort results by f1_score
        results_list = sorted(results_list, key=lambda x: x[5], reverse=True)

        # Show results
        print()
        print('(EVALUATOR) Sorted results: ')
        for vectorizer, model, accuracy, precision, recall, f1_score in results_list:
            print()
            print("Model - ", model)
            print("Vectorizer - ", vectorizer)            
            print("-> F1 Score - ", "{0:.2f}".format(f1_score))
            print("-> Precision - ", "{0:.2f}".format(precision))
            print("-> Recall - ", "{0:.2f}".format(recall))
            print("-> Accuracy - ", "{0:.2f}".format(accuracy))

        # Pick best model, train it and save it
        model = Model(vectorization=results_list[0][0], model=results_list[0][1])
        model.train()
        model.save()

        print()
        print('(EVALUATOR) Trained and saved best model')

    # Soft evaluation: Just print model evaluations
    else:

        for model_type in const.MODELS:

            print()

            # 1. Create model
            model = Model(model=model_type, evaluation=evaluation)
            print('(EVALUATOR) Model ' + model_type + ' created')

            # 2. Train classifier
            model.train(grid_search=main_grid_search)
            print('(EVALUATOR) Model ' + model_type + ' trained')

            # 3. Evaluate classifier
            accuracy, _, report, matrix = model.evaluate()
            print('(EVALUATOR) Model ' + model_type + ' evaluated')
            print()
            print('(EVALUATOR) Evaluation, Accuracy: ' + str(accuracy))
            print('(EVALUATOR) Evaluation, Classification Report: ')
            print(report)
            print()
            print('(EVALUATOR) Evaluation, Confusion Matrix: ')
            print(matrix)
            print()

if __name__ == "__main__":

    main_hard_evaluation = int(sys.argv[1]) == 1 # 0 no, 1 yes
    main_cross_evaluation = int(sys.argv[2]) # 1 normal, 2 cross
    main_grid_search = int(sys.argv[3]) == 1 # 0 no, 1 yes

    evaluate(main_hard_evaluation, main_cross_evaluation, main_grid_search)
