# Introducción al Procesamiento de Lenguaje Natural - 2019
Laboratorios del curso de Introducción al Procesamiento de Lenguaje Natural 2019, Facultad de Ingeniería (UdelaR)

**Laboratorio - Clasificador de tweets humorísticos**

# Dependencias
Para la ejecución de este laboratorio se empleó:
- Python v3.7.4
- sckit-learn v0.21.3
- keras v2.3.1
- pandas v0.25.2
- numpy v1.8.2
- pickle v4.0
- NLTK v3.4.5

# Interfaces
Se proporcionan las siguientes interfaces:
- `es_humor.py` - Intefaz requerida por el laboratorio.
  - Ejemplo de invocación: `python3 es_humor.py data/ data_test.csv`
- `src/classifier.py` - Interfaz que carga el último modelo entrenado y clasifica un tweet ingresado mediante input.
- `src/trainer.py` - **TODO: Borrar esto?**
- `src/evaluator.py` - Interfaz que 
  - Invocación: python3 evaluator.py hard_evaluation cross_evaluation grid_search **TODO: Borrar cross eval?**
  - En donde: hard_evaluator = (0|1), cross_evaluation = (1|2), grid_search = (0|1). Ejemplo: `python3 evaluator.py 0 1 0`
  - Hard evaluator: Si esta flag está en 1 se realiza una `Hard evaluation`, de lo contrario se realiza una `Soft Evaluation`
    - Hard evaluation: Los modelos se entrenan secuencialmente, imprimiendo sus resultados de forma ordenada según cual alcanzó mejor F-measure.
    - Soft evaluation: Los modelos se entrenan secuencialmente, imprimiendo sus resultados conforme sean entrenados.
  - Grid Search: Si esta flag está en 1 se realiza una `Grid Search`, de lo contrario se entrena usando parámetros por defecto.
    - Esta flag cambia los modelos de `Scikit-learn` por un `GridSearchCV` que engloba cada respectivo modelo, esto permite que la función .fit se ejecute en una grilla de hiperparámetros, empleando validación cruzada en tres partes para determinar la mejor combinación de hiperparámetros.
    - Los parametros con los que se experimentó se listan en `src/model.grid_search_params`

# Herramientas
Para el procesamiento de los datos se implementaron las siguientes herramientas:
- `src/util/corrector.py` - **TODO: qué mierda hacemos con esto?**
- `src/vectorization/embeddings_vectorizer` - Vectoriza un tweet a una lista de floats (procedimiento que será detallado a continuación).
- `src/vectorization/features_vectorizer` - Vectoriza un tweet a una lista de features (las cuales serán detalladas a continuación).
- `src/vectorization/vectorizer` - Interfaz que abstrae el uso que el modelo hace del vectorizer.

# Metodología
Para el procesamiento de los tweets se proporcionan dos estratégias, las cuales consisten en:

## Features
Basadas en el proyecto de grado [Detección de humor en textos en español](https://www.fing.edu.uy/inco/grupos/pln/prygrado/Informepghumor.pdf) [GitHub](https://github.com/pln-fing-udelar/pghumor), desarrollado por Santiago Castro y Matías Cubero.

Esta estrategia vectoriza un tweet a una lista de caracteristicas inducidas a partir del mismo tweet.

Primero el texto del tweet es procesado a minusculas, y se le substituyen los simbolos de espaciado por un sólo espacio en blanco.

Luego a partir del texto procesado se extraen las siguientes features:
- `starts_with_dialogue` - [0-1], indica si el texto comienza con un guión de dialogo.
- `number_of_urls` - Entero, cuenta el número de enlaces contenidos dentro del texto.
- `number_of_exclamations` - Entero, cuenta el número de simbolos de exclamación contenidos dentro del texto.
- `number_of_hashtags` - Entero, cuenta el número de hastags contenidos dentro del texto.
- `number_of_question_answers` - Entero, cuenta el número de oraciones englobadas en `¿?` o que terminan en `?` seguidas de una oración no englobada o seguida por simbolos de pregunta.
- `ratio_of_keywords` - Real no negativo, ratio de ocurrencias de palabras clave en el total de palabras del tweet. Las palabras claves empleadas se situan en `src/vectorization/dictionaries/keywords.dic`
- `ratio_of_animals` - Real no negativo, ratio de ocurrencias de animales en el total de palabras del tweet. Los nombres de animales empleados se situan en `src/vectorization/dictionaries/animales.dic`
- `ratio_of_sexual_words` - Real no negativo, ratio de ocurrencias de palabras sexuales en el total de palabras del tweet. Las palabras sexuales empleadas se situan en `src/vectorization/dictionaries/sexual.dic`
- `capslock_ratio` - Real no negativo, ratio de ocurrencias de palabras en mayuscula en el total de palabras del tweet.

Es a partir de estas caracteristicas que se espera los modelos sean capaces de inferir si el tweet procesado era un chiste o no.

## Word Embeddings
Esta estrategia hace uso del archivo de word embeddings entregado para el laboratorio.

Primero el texto del tweet es procesado, eliminando simbolos no deseados de puntuación `[., , , _,;,",\n,',!,:,?]`.

A continuación el texto es tokenizado mediante la función `tokenize` del modelo `tokenizer` de NLTK.

Por último a partir del vector de tokens se construye un vector de longitud 300, compuesto en cada posición por el promedio de los word embeddings de cada token en el tweet.

Se espera que este word embedding refleje en cierto grado la semántica del tweet y permita que los modelos infieran si el tweet procesado era un chiste o no.

# Experimentación
Para cada una de las estrategias mencionadas anteriormente, se realizó un grid search a los siguientes modelos de (scikit learn)[https://scikit-learn.org/]:
- KNeighborsClassifier
- Support Vector Classifier
- DecisionTreeClassifier 
- MultinomialNB 
- MLPClassifier

Adicionalmente se entrenó una red de (Keras)[https://keras.io/], haciendo uso de la capa de embeddings proporcionada por keras.
**TODO: Germán escribir detalles de la red**

# Conclusión
Tras procesar los resultados de la experimentación, se dio con la mejor solución para el problema **INSERTAR SOLUCIÓN AQUÍ**, dicha solución es proporcionada en `es_humor.py`, a modo de ser evaluada en el laboratorio.