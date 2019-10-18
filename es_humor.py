# python3 es_humor.py <data_path> test_file1.csv … test_fileN.csv
# Donde <data_path> tiene la ruta donde se encuentran los datos impartidos:
#   • humor_train.csv, humor_val.csv, humor_test.csv
#   • intropln2019_embeddings_es_300.txt
#   y test_file1.csv … test_fileN.csv son un conjunto de archivos de test (pueden tener salida o no).

# El programa debe:
# 1. Entrenar un clasificador (hiperaparámetros encontrados previamente) utilizando los conjuntos de train y val (debe demorar menos de 10 minutos en una CPU intel i7)
# 2. Por cada archivo test_file1.csv … test_fileN.csv el programa debe:
#    1. aplicar el modelo previamente entrenado
#    2. generar un archivo de salida test_file1.out … test_fileN.out con las salidas obtenidas a cada archivo de test.
#       El archivo de salida debe tener las salidas (0 o 1) en orden y separados por un fin de línea. (Ej. 1\n0\n0\n1...\n0)

import src.trainer as trainer
import pandas as pd

def read_input():
    if len(sys.argv) < 2:
        raise Exception('Cantidad insuficiente de parametros')
    data_path = sys.argv[1]
    test_files = []
    for test_file in sys.argv[:2]:
        test_files += test_file

def test(test_file):
    df_test = pd.read_csv(data_path + test_file)
    prediction = model.predict(df_test['text'].values.astype('U'))

    accuracy = accuracy_score(df_test['humor'].values, prediction)
    results = classification_report(df_test['humor'].values, prediction, output_dict=True)
    matrix = confusion_matrix(df_test['humor'].values, prediction)

    report = {
        'f1_score': results['macro avg']['f1-score'],
        'precision': results['macro avg']['precision'],
        'recall': results['macro avg']['recall'],
    }
    print('Results:')
    print()
    print('Accuracy: ' + str(accuracy))
    print('Classification Report: ')
    print(report)
    print()
    print('Confusion Matrix: ')
    print(matrix)
    print()

    f = open(f"{test_file.replace('.csv', '')}.out", "a")
        for pred in predictions[:-1]:
            f.write(f"{pred}\n")
        f.write(f"{predictions[-1]}")
        f.close()

def main():
    read_input()

    model = trainer.train(data_path=data_path, best_solution=True)

    for test_file in test_files:
        test(test_file)
  
main()