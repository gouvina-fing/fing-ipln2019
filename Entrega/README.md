# Introducción al Procesamiento de Lenguaje Natural - 2019
Laboratorios del curso de Introducción al Procesamiento de Lenguaje Natural 2019, Facultad de Ingeniería (UdelaR)

**Laboratorio 1 - Clasificador de tweets humorísticos**


# Modulos y versiones utilizados: 
Del modulo  sklearn.svm se importa SVC
Del modulo  sklearn.tree se importa DecisionTreeClassifier
Del modulo  sklearn.naive_bayes se importa MultinomialNB
Del modulo  sklearn.neighbors se importa KNeighborsClassifier
Del modulo  sklearn.neural_network se importa MLPClassifier
Del modulo  sklearn.model_selection se importa StratifiedKFold, cross_validate, GridSearchCV
Del modulo  sklearn.feature_extraction.text se importa CountVectorizer
Del modulo  sklearn.feature_extraction se importa DictVectorizer
Del modulo  sklearn.metrics se importa accuracy_score, classification_report, confusion_matrix
Del modulo  vectorization.vectorizer se importa Vectorizer


    
Durante el proyecto se evaluarion 5 modelos de clasificacion :
- KNeighborsClassifier 
- Support Vector Classifier
- DecisionTreeClassifier 
- MultinomialNB 
- MLPClassifier
- Keras

- ## Tokenizacion de tweets:
    Para la tokenizacion de un tweet primero se deven preprocesar los datos eliminado simbolos no deseados `[., , , _,;,",\n,',!,:,?]`, luego se utiliza la funcion  tokenize (del modulo tokenizer) aplicada al tweet.

**Funcionamiento de Keras:**
    
- ## Calculo de vector 300 asociado a un tweets:
    Dado un tweet el mismo es desconpuesto mediante un proceso de tokenizacion, luego se busca el vector de cada uno de los tokens en el archivo embeddings en el caso de no contener alguno de los tokens a ese mismo se le asigna el vector cuyas entradas son 0 en cada una de sus 300 posiciones, una vez obtenidos todos los vectores de los tokens que componen el tweet se calcula el vector promedio y es este el que se utiliza como el vector asociado al tweet.
- ## Uso de vector 300 asociado a un tweets:
    Una vez obtenido el vector asociado a un tweet a evaluar se puede calcular de distancia entre el mismo y un tweet previamente clasificado. Utilizando la cercania de un Tweet a evaluar a otros se puede determinar si se trata de un chiste o de un no chiste.
    
# Modelos
## Proceso aplicado
- **Preproceso de datos**
