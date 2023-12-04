from sklearn.model_selection import train_test_split
import numpy as np

from common.utils import get_dict_of_lists, get_cv
from common.metrics import get_metrics
from models.models import build_model
from models.hyperopt import CVTuner

NUM_CV_SPLITS = 10
TUNING_DATA_SHARE = 0.25
BATCH_SIZE = 16


def search_hp(X, y, demo, model, method, set_params, objective, max_trials, project_name='untitled_project'):
    tuner = CVTuner(demo, model, method, set_params, objective, max_trials, project_name, BATCH_SIZE)
    tuner.search(X, y)
    return tuner.get_best_hyperparameters(1)[0].values, tuner.losses


def cross_validate(X, y, demo, model, method, params):
    def iterate_cv(train_idx, test_idx):
        train_data = {'img': X[train_idx]}
        test_data = {'img': X[test_idx]}
        if 'multiinput' in model:
            train_data['demo'] = train_demo[train_idx]
            test_data['demo'] = train_demo[test_idx]
        builder = build_model[model][method]
        model_instance = builder(**params)
        history = model_instance.fit(train_data, y[train_idx], BATCH_SIZE, params['epochs'], verbose=0)
        y_pred = model_instance.predict(test_data, verbose=0)
        metrics = get_metrics[method](y[test_idx], y_pred)
        return metrics, history.history, (y[test_idx], y_pred)

    cv = get_cv[method](n_splits=NUM_CV_SPLITS)
    train_demo = np.hstack((demo[:, 3:], np.reshape(1 - demo[:, -1], (demo.shape[0], 1)))).astype('float32')
    scores, histories, predictions = zip(*[iterate_cv(*idx) for idx in cv.split(X, y, demo[:, 0])])
    scores = get_dict_of_lists(scores)
    return scores, histories, predictions


def kt_search(X, y, demo, model, method, set_params, objective, max_trials=50, project_name='untitled_project'):
    stratify = y if method == 'clf' else None
    X_tune, X_val, y_tune, y_val, demo_tune, demo_val = train_test_split(X, y, demo, stratify=stratify, train_size=TUNING_DATA_SHARE)
    params, tune_history = search_hp(X_tune, y_tune, demo_tune, model, method, set_params, objective, max_trials, project_name)
    cv_scores, history, predictions = cross_validate(X_val, y_val, demo_val, model, method, params)
    return params, cv_scores, predictions, history, tune_history
