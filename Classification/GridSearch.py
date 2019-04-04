import sys
from pprint import pprint
from time import time

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from Classification import Classifier, ClassifierType
from Utils import DataStructures, Log


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


class GridSearch(object):
    """
    Exhaustive search over specified parameter values for an estimator.
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    classifier: Classifier

    def __init__(self, classifier_type: ClassifierType):
        self.classifier = Classifier.factory(classifier_type)
        self.parameters = self.get_parameters()

    def _get_classifier_params(self):
        params = {}
        for key, value in self.classifier.get_search_params().items():
            param_key = f"{self.classifier.get_short_name()}__{key}"
            params[param_key] = value

        return params

    @staticmethod
    def _get_count_vectorizer_params():
        return {
            # 'vect__max_df': [0.5, 0.75, 1.0],
            # 'vect__max_features': (None, 5000, 10000, 50000),
            # 'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],  # unigrams or bigrams
            'vect__max_df': [1.0],
            'vect__ngram_range': [(1, 1)],  # unigrams or bigrams
        }

    @staticmethod
    def _get_tfidf_transformer_params():
        return {
            'tfidf__sublinear_tf': [False, True],
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
        }

    def get_parameters(self):
        vectorizer_parameters = self._get_count_vectorizer_params()
        transformer_params = self._get_tfidf_transformer_params()
        classifier_params = self._get_classifier_params()

        self.parameters = DataStructures.merge_dicts(vectorizer_parameters, transformer_params, classifier_params)
        return self.parameters

    def fit(self, training_data, training_labels, n_jobs: int = -1, dense_data: bool = False):
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            (self.classifier.get_short_name(), self.classifier.classifier),
        ])

        if dense_data:
            dense_transformer = ('to_dense', DenseTransformer())
            pipeline.steps.insert(2, dense_transformer)
            Log.info("Added dense data transformer to pipeline.")

        grid_search = GridSearchCV(pipeline, self.parameters, cv=5, n_jobs=n_jobs, verbose=1)

        Log.info(f"Performing grid search on {self.classifier.get_name()}...", header=True)
        Log.info("pipeline:", [name for name, _ in pipeline.steps])
        Log.info("parameters:")
        pprint(self.parameters)
        t0 = time()
        grid_search.fit(training_data, training_labels)
        Log.info("done in %0.3fs" % (time() - t0))

        Log.info("Results", header=True)
        pprint(grid_search.cv_results_)

        Log.info("Best score: %0.3f" % grid_search.best_score_, header=True)
        Log.info("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(self.parameters.keys()):
            Log.info("\t%s: %r" % (param_name, best_parameters[param_name]))
