(EVALUATOR) Model svm created
Best hyper parameters for svm are: (Score: 0.7597505735631508)

{'C': 100, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True}

SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)


(EVALUATOR) Model svm trained
(EVALUATOR) Model svm evaluated

(EVALUATOR) , Accuracy: 0.7884
(EVALUATOR) , Classification Report: 
              precision    recall  f1-score   support

           0       0.78      0.95      0.86      3312
           1       0.83      0.47      0.60      1688

    accuracy                           0.79      5000
   macro avg       0.80      0.71      0.73      5000
weighted avg       0.80      0.79      0.77      5000


(EVALUATOR) , Confusion Matrix: 
[[3152  160]
 [ 898  790]]


(EVALUATOR) Model tree created
Best hyper parameters for tree are: (Score: 0.7658685805047356)

{'criterion': 'gini', 'max_depth': 8}

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')


(EVALUATOR) Model tree trained
(EVALUATOR) Model tree evaluated

(EVALUATOR) , Accuracy: 0.7904
(EVALUATOR) , Classification Report: 
              precision    recall  f1-score   support

           0       0.78      0.94      0.86      3312
           1       0.82      0.49      0.61      1688

    accuracy                           0.79      5000
   macro avg       0.80      0.72      0.73      5000
weighted avg       0.79      0.79      0.77      5000


(EVALUATOR) , Confusion Matrix: 
[[3126  186]
 [ 862  826]]


(EVALUATOR) Model nb created
/home/renzodgc/env/ipln2019/venv/lib/python3.6/site-packages/sklearn/naive_bayes.py:485: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  'setting alpha = %.1e' % _ALPHA_MIN)
/home/renzodgc/env/ipln2019/venv/lib/python3.6/site-packages/sklearn/naive_bayes.py:485: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  'setting alpha = %.1e' % _ALPHA_MIN)
/home/renzodgc/env/ipln2019/venv/lib/python3.6/site-packages/sklearn/naive_bayes.py:485: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
  'setting alpha = %.1e' % _ALPHA_MIN)
Best hyper parameters for nb are: (Score: 0.7471615977410436)

{'alpha': 2.0}

MultinomialNB(alpha=2.0, class_prior=None, fit_prior=True)


(EVALUATOR) Model nb trained
(EVALUATOR) Model nb evaluated

(EVALUATOR) , Accuracy: 0.777
(EVALUATOR) , Classification Report: 
              precision    recall  f1-score   support

           0       0.76      0.98      0.85      3312
           1       0.89      0.39      0.54      1688

    accuracy                           0.78      5000
   macro avg       0.82      0.68      0.70      5000
weighted avg       0.80      0.78      0.75      5000


(EVALUATOR) , Confusion Matrix: 
[[3233   79]
 [1036  652]]


(EVALUATOR) Model knn created
Best hyper parameters for knn are: (Score: 0.7581622448379316)

{'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='uniform')


(EVALUATOR) Model knn trained
(EVALUATOR) Model knn evaluated

(EVALUATOR) , Accuracy: 0.7776
(EVALUATOR) , Classification Report: 
              precision    recall  f1-score   support

           0       0.78      0.92      0.85      3312
           1       0.76      0.50      0.60      1688

    accuracy                           0.78      5000
   macro avg       0.77      0.71      0.73      5000
weighted avg       0.77      0.78      0.76      5000


(EVALUATOR) , Confusion Matrix: 
[[3037  275]
 [ 837  851]]


(EVALUATOR) Model mlp_classifier created
Best hyper parameters for mlp_classifier are: (Score: 0.7686334490264133)

{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'max_iter': 2000, 'solver': 'adam'}

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(50, 50, 50), learning_rate='constant',
              learning_rate_init=0.001, max_iter=2000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=None, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


(EVALUATOR) Model mlp_classifier trained
(EVALUATOR) Model mlp_classifier evaluated

(EVALUATOR) , Accuracy: 0.7942
(EVALUATOR) , Classification Report: 
              precision    recall  f1-score   support

           0       0.78      0.95      0.86      3312
           1       0.84      0.48      0.61      1688

    accuracy                           0.79      5000
   macro avg       0.81      0.72      0.74      5000
weighted avg       0.80      0.79      0.78      5000


(EVALUATOR) , Confusion Matrix: 
[[3155  157]
 [ 872  816]]