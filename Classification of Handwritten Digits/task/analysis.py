# write your code here
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

new_X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

X_new = pd.DataFrame(new_X_train[:6000, :])
y_new = pd.Series(y_train[:6000])

scores_stage_3 = dict()
scores_stage_4 = dict()


def stage_1():
    print(f"Classes: {np.unique(y_train)}")
    print(f"Features' shape: {new_X_train.shape}")
    print(f"Target's shape: {y_train.shape}")
    print(f"min: {new_X_train.min()}, max: {new_X_train.max()}")


def stage_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=40)

    print(f"x_train shape: {X_train.shape}")
    print(f"x_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Proportion of samples per class in train set:")
    # normalize=True to return proportion instead of frequency
    print(y_train.value_counts(normalize=True).round(2))


# stage_2(X_new, y_new)


def stage_3(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    def fit_predict_eval(Model, features_train, features_test, target_train, target_test):
        # here you fit the model
        model = Model()
        model.fit(features_train, target_train)
        # make a prediction
        predictions = model.predict(features_test)
        # calculate accuracy and save it to score
        score = accuracy_score(target_test, predictions)
        scores_stage_3.update({Model.__name__: score.__round__(3)})
        # print(f'Model: {model}\nAccuracy: {score}\n')

    # Test for all models to find best solution
    # Models: K-nearest Neighbors, Decision Tree, Logistic Regression, and Random Forest.
    Models = [KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier]

    for Model in Models:
        fit_predict_eval(
            Model=Model,
            features_train=X_train,
            features_test=X_test,
            target_train=y_train,
            target_test=y_test
        )

    max_score = max(scores_stage_3.values())
    max_score_name = [key for key, val in scores_stage_3.items() if val == max_score][0]
    # print(f"The answer to the question: {max_score_name} - {max_score}")


# stage_3(X_new, y_new)


def stage_4(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    normalize = Normalizer()
    X_train_norm = normalize.fit_transform(X_train)
    X_test_norm = normalize.fit_transform(X_test)

    def fit_predict_eval(Model, features_train, features_test, target_train, target_test):
        # here you fit the model
        model = Model()
        model.fit(features_train, target_train)
        # make a prediction
        predictions = model.predict(features_test)
        # calculate accuracy and save it to score
        score = accuracy_score(target_test, predictions)
        scores_stage_4.update({Model.__name__: score.__round__(3)})
        print(f'Model: {model}\nAccuracy: {score.__round__(3)}\n')

    # Test for all models to find best solution
    # Models: K-nearest Neighbors, Decision Tree, Logistic Regression, and Random Forest.
    Models = [KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier]

    for Model in Models:
        fit_predict_eval(
            Model=Model,
            features_train=X_train_norm,
            features_test=X_test_norm,
            target_train=y_train,
            target_test=y_test
        )

    top_scores = list(scores_stage_4.values())
    model_names = list(scores_stage_4.keys())
    top_score_1, top_score_2 = sorted(top_scores, reverse=True)[:2]
    top_index_1, top_index_2 = top_scores.index(top_score_1), top_scores.index(top_score_2)

    print(f"The answer to the 1st question: yes")
    print(f"The answer to the 2nd question: {model_names[top_index_1]}-{top_score_1}, "
          f"{model_names[top_index_2]}-{top_score_2}")


# stage_4(X_new, y_new)

def stage_5(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    normalize = Normalizer()
    X_train_norm = normalize.fit_transform(X_train)
    X_test_norm = normalize.fit_transform(X_test)

    def fit_predict_eval(model, features_train, features_test, target_train, target_test):
        # here you fit the model
        model.fit(features_train, target_train)
        # make a prediction
        predictions = model.predict(features_test)
        # calculate accuracy and save it to score
        score = accuracy_score(target_test, predictions)
        print(f'best estimator: {model}\nAccuracy: {score.__round__(4)}\n')

    # For KNeighborsClassifier

    def my_KNeighborsClassifier():
        estimator = KNeighborsClassifier()
        param_grid = {
            "n_neighbors": [3, 4],
            "weights": ['uniform', 'distance'],
            "algorithm": ['auto', 'brute']
        }
        clf = GridSearchCV(estimator=estimator,
                           param_grid=param_grid,
                           scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)

        best_params = clf.best_params_

        model = KNeighborsClassifier(
            n_neighbors=best_params["n_neighbors"],
            weights=best_params["weights"],
            algorithm=best_params["algorithm"],
        )

        fit_predict_eval(
            model,
            X_train_norm,
            X_test_norm,
            y_train,
            y_test,
        )
        sorted(clf.cv_results_.keys())

    # For RandomForestClassifier        print(model)

    def my_RandomForestClassifier():
        estimator = RandomForestClassifier()
        param_grid = {
            "n_estimators": [300, 500, 700],
            "max_features": ['sqrt', 'log2'],
            "class_weight": ['balanced', 'balanced_subsample'],
            "random_state": [40],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy", "log_loss"],
        }
        clf = GridSearchCV(estimator=estimator,
                           param_grid=param_grid,
                           scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)

        best_params = clf.best_params_

        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            class_weight=best_params["class_weight"],
            random_state=best_params["random_state"],
            bootstrap=best_params["bootstrap"],
            criterion=best_params["criterion"],
        )

        fit_predict_eval(
            model,
            X_train_norm,
            X_test_norm,
            y_train,
            y_test,
        )
        sorted(clf.cv_results_.keys())

    print("K-nearest neighbours algorithm")
    my_KNeighborsClassifier()

    print("Random forest algorithm")
    my_RandomForestClassifier()


stage_5(X_new, y_new)
