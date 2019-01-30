from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

sns.set(style="ticks")

os.chdir("C:/Users/AE250016/Desktop/ACA_DS/Untitled Folder")

titanic = pd.read_csv('train.csv')
titanic = titanic[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


class ModelingStage:

    @staticmethod
    def dict_model(inputs):
        """ Handles dictionary of Sk Learn models

        Args:
            inputs (String or List or Dict):
            1) String with required model name
            2) List of required models
            3) Dictionary with tuples (model() , {Parameter dictionary} )

        Returns: Dictionary with tuples (model() , {Parameter dictionary} )

        """

        dictionary = {"Trees": (DecisionTreeClassifier(), {'max_depth': np.arange(3, 10)}),
                      "Logistic": (LogisticRegression(), {'C': [0.001, 0.01, 0.05, 0.1, 10, 100]}),
                      'K-nearest-neighbour': (KNeighborsClassifier(),
                                              {'n_neighbors': [5, 6, 7, 8, 9],
                                               'metric': ['minkowski', 'euclidean', 'manhattan'],
                                               'weights': ['uniform', 'distance']})}
        if inputs:
            if isinstance(inputs, dict):
                return inputs
            elif isinstance(inputs, str):
                filtered_dictionary = {inputs: dictionary[inputs]}
                return filtered_dictionary
            elif isinstance(inputs, list):
                filtered_dictionary = {}
                for a in inputs:
                    filtered_dictionary[a] = dictionary[a]
                return filtered_dictionary
        else:
            return dictionary

    def plot_learning_curve(self, loading_eda, scores='neg_log_loss'):
        """

        Args:
            loading_eda (class.LoeadingEDA ):  Object of Loeading EDA class
            scores (String): Type of scoring

        Returns:
            Plot with learning curves

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.figure()
            plt.title('title')
            plt.xlabel("Training examples")
            plt.ylabel(scores)
            model = self.best_model.fit(loading_eda.X_train, loading_eda.y_train)
            train_sizes, train_scores, test_scores = learning_curve(model, loading_eda.x_train, loading_eda.y_train,
                                                                    cv=5, scoring=scores)
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            plt.grid()
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
            plt.legend(loc="best")

    @staticmethod
    def dummy_model(x_train, y_train, x_test, y_test, dummy_strategy):
        """ calculates accuracy score of the dummy model on test data

        Args:
            x_train(numpy array): for training data
            y_train(numpy array): for training target labels
            x_test(numpy array): for testing data
            y_test(numpy array): for testing target labels
            dummy_strategy (String): type of dummy model to use

        Returns: accuracy score of the dummy model on test data

        """
        dummy_model = DummyClassifier(strategy=dummy_strategy).fit(x_train, y_train)
        y_dummy = dummy_model.predict(x_test)
        return accuracy_score(y_test, y_dummy)

    @staticmethod
    def modeling_stage_k_folds(model_dictionary, x_train, y_train, k_folds, performance_metric):
        """ Choosing the best model applying cross_fold validation

        Args:
            model_dictionary: Dictionary with tuples( model(), {Parameter dictionary})
            x_train(numpy array): for training data
            y_train(numpy array): for training target labels
            k_folds (int):  Number of cross folds
            performance_metric (String):  Metric to be used

        Returns:
            model_dicts (dict): Dictionary with best accuracy per medel as key, and the model as value
            cross_val_results (pd.Dataframe):
            best_model (Sklearn.Model):

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cross_val_results = pd.DataFrame()
            model_dicts = {}

            for a in model_dictionary.keys():
                grid_clf_acc = GridSearchCV(model_dictionary[a][0].fit(x_train, y_train),
                                            param_grid=model_dictionary[a][1],
                                            scoring=performance_metric,
                                            cv=k_folds)

                grid_clf_acc = grid_clf_acc.fit(x_train, y_train)

                temp = pd.DataFrame(grid_clf_acc.cv_results_)
                temp['model'] = a
                cross_val_results = cross_val_results.append(temp)

                model_dicts[grid_clf_acc.best_score_] = grid_clf_acc.best_estimator_
            cross_val_results = cross_val_results.set_index(['model'], append=True)
            best_model = model_dicts[max(model_dicts.keys())]
            return model_dicts, cross_val_results, best_model

    @staticmethod
    def modeling_validation(dictionary, x_train, y_train, x_val, y_val, scoring='accuracy'):
        """

        Args:
        dictionary (dict): Dictionary with tuples( model(), {Parameter dictionary})
        x_train(numpy array): for training data
        y_train(numpy array): for training target labels
        x_val(numpy array): for validation data
        y_val(numpy array): for validation testing data
        scoring(String):  Metric to be used

        Returns:
         model_dicts (dict): Dictionary with best accuracy per medel as key, and the model as value
         cross_val_results (pd.Dataframe):
         best_model (Sklearn.Model):

        """
        def score_best_param_per_model(rf, grid, model_type, scoring):
            best_score = 0
            classifier_results = pd.DataFrame()
            scorer = get_scorer(scoring)
            for g in ParameterGrid(grid):
                rf.set_params(**g)
                rf.fit(x_train, y_train)
                # save if best
                if scorer(X=x_val, estimator=rf, y_true=y_val) > best_score:
                    best_score = scorer(X=x_val, estimator=rf, y_true=y_val)
                    best_grid = g
                temp = pd.DataFrame({model_type: g}).transpose()
                temp['val_score'] = scorer(X=x_val, estimator=rf, y_true=y_val)
                temp['train_score'] = scorer(X=x_train, estimator=rf, y_true=y_train)
                classifier_results = classifier_results.append(temp)
                best_estimator = rf.set_params(**best_grid)
            return best_score, best_grid, classifier_results.sort_values('val_score', ascending=False), best_estimator

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_results = pd.DataFrame()
            model_dicts = {}
            for a in dictionary.keys():
                best_score, best_grid, classifier_results, best_estimator = score_best_param_per_model(dictionary[a][0],
                                                                                                       dictionary[a][1],
                                                                                                       a, scoring)
                val_results = val_results.append(classifier_results)
                model_dicts[best_score] = best_estimator
            best_model = model_dicts[max(model_dicts.keys())]
        return model_dicts, val_results.sort_values('val_score', ascending=False), best_model

    def __init__(self,
                 loading_eda,
                 k_folds=10,
                 performance_metric='accuracy',
                 dummy_strategy='stratified',
                 inputs=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dictionary = ModelingStage.dict_model(inputs)
            if isinstance(k_folds, float):
                self.best_accuracy_per_model,\
                    self.cv_results,\
                    self.best_model = self.modeling_validation(dictionary,
                                                               loading_eda.x_train,
                                                               loading_eda.y_train,
                                                               loading_eda.x_val,
                                                               loading_eda.y_val,
                                                               performance_metric)
            else:
                self.best_accuracy_per_model,\
                    self.cv_results,\
                    self.best_model = self.modeling_stage_k_folds(dictionary,
                                                                  loading_eda.x_train,
                                                                  loading_eda.y_train,
                                                                  k_folds,
                                                                  performance_metric)
            self.dummy_accuracy = self.dummy_model(loading_eda.x_train, loading_eda.y_train, loading_eda.x_test,
                                                   loading_eda.y_test, dummy_strategy)
            self.performance_metric = performance_metric