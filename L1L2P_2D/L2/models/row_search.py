from sklearn.model_selection import cross_validate, GridSearchCV, KFold, PredefinedSplit
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from default_values import *


def best_search_params(search_results):
    search_params = search_results['params'][0].keys()
    best_comb = np.argmax(search_results['mean_test_score'])
    best_params = search_results['params'][best_comb]
    best_dict = {}
    for param in search_params:
        best_dict[param.removeprefix('model__')] = best_params[param]
    return best_dict


def get_custom_cv(n_splits, value_indices=None):
    if value_indices is None:
        return KFold(n_splits=n_splits)
    else:
        return PredefinedSplit(value_indices // n_splits)


def get_search_params(X, y, search, model, params, value_indices=None, n_jobs=4, n_splits=3):
    pipeline = search(
        Pipeline([
            ('scaler', StandardScaler()),
            ('model', model()),
        ]),
        dict(zip([f'model__{param}' for param in params], list(params.values()))),
        cv=get_custom_cv(n_splits, value_indices), n_jobs=n_jobs
    )
    pipeline.fit(X, y)
    search_results = pipeline.cv_results_
    best_params = best_search_params(search_results)
    return best_params


def get_cv_score(X, y, model, params, scorers, value_indices=None, n_jobs=4, cv_folds=10):
    cv_scores = cross_validate(model(**params), X, y, cv=get_custom_cv(cv_folds, value_indices),
                               scoring=scorers, n_jobs=n_jobs)
    for name, value in cv_scores.items():
        cv_scores[name] = (np.round(np.mean(value), 3), np.round(np.std(value), 3))
    return cv_scores


search_classes = {
    'Grid': GridSearchCV,
    'Bayes': BayesSearchCV
}


def search_cv(X, y, model_name, method, search, value_indices=None, params=None, n_jobs=4, debug=False):
    if debug:
        params = test_params[search][model_name]
    elif params is None:
        params = default_params[search][model_name]
    model = default_models[method][model_name]
    scorers = default_scorers[method]
    search_class = search_classes[search]

    search_results = get_search_params(X, y, search_class, model, params, value_indices, n_jobs=n_jobs)
    cv_results = get_cv_score(X, y, model, search_results, n_jobs=n_jobs, scorers=scorers, value_indices=None)
    return search_results, cv_results
