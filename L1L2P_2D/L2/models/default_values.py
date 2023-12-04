from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
                             RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from skopt.space import Real, Categorical, Integer

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
                            r2_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import label_binarize

knn_n_neighbors = list(range(1, 11))
knn_p_norm = [x / 20 for x in range(20, 101)]  # [1, 5] range with 0.05-sized steps

mlp_neurons = list(range(10, 201, 10))
mlp_max_iter = list(range(1000, 10000, 1000)) + list(range(10000, 50001, 5000))
mlp_lr = [10 ** i for i in range(-6, -1)]
mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
mlp_solver = ['sgd', 'adam']

rfc_estimators = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))
rfc_min_samples_split = list(range(2, 11))
rfc_min_samples_leaf = list(range(1, 11))
gbc_lr = [10 ** (-3), 10 ** (-2)] + [i / 10 for i in range(1, 6)]

default_gs_params = {
    'knn': {
        'n_neighbors': knn_n_neighbors,
        'p': knn_p_norm,
    },
    'mlp': {
        'hidden_layer_sizes': mlp_neurons,
        'max_iter': mlp_max_iter,
        'learning_rate_init': mlp_lr,
        'activation': mlp_activation,
        'solver': mlp_solver,
    },
    'ab': {
        'learning_rate': gbc_lr,
        'n_estimators': rfc_estimators,
    },
    'rf': {
        'n_estimators': rfc_estimators,
        'min_samples_split': rfc_min_samples_split,
        'min_samples_leaf': rfc_min_samples_leaf,
    },
    'gb': {
        'learning_rate': gbc_lr,
        'n_estimators': rfc_estimators,
        'min_samples_split': rfc_min_samples_split,
        'min_samples_leaf': rfc_min_samples_leaf,
    }
}
default_bs_params = {
    'knn': {
        'n_neighbors': Integer(1, 10),
        'p': Real(1, 5),
    },
    'mlp': {
        'hidden_layer_sizes': Integer(2, 200, prior='log-uniform'),
        'max_iter': Integer(10, 50000, prior='log-uniform'),
        'learning_rate_init': Real(10 ** (-6), 10 ** (-2), prior='log-uniform'),
        'activation': Categorical(mlp_activation),
        'solver': Categorical(mlp_solver),
    },
    'ab': {
        'learning_rate': Real(10 ** (-3), 0.5, prior='log-uniform'),
        'n_estimators': Integer(10, 10000, prior='log-uniform'),
    },
    'rf': {
        'n_estimators': Integer(10, 10000, prior='log-uniform'),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    },
    'gb': {
        'learning_rate': Real(10 ** (-3), 0.5, prior='log-uniform'),
        'n_estimators': Integer(10, 10000, prior='log-uniform'),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    },
    'basic_cnn': {
        'conv1_size': Integer(8, 16),
        'conv2_size': Integer(16, 32),
        'conv3_size': Integer(32, 64),
        'dropout_rate': Real(0.1, 0.8),
        'dense_size': Integer(64, 128),
    }
}


default_params = {
    'Grid': default_gs_params,
    'Bayes': default_bs_params
}

test_gs_params = {
    'knn': {
        'n_neighbors': [10],
        'p': [2.5],
    },
    'mlp': {
        'hidden_layer_sizes': [100],
        'max_iter': [10000],
        'learning_rate_init': [10 ** (-3)],
        'activation': ['relu'],
        'solver': ['adam'],
    },
    'ab': {
        'learning_rate': [0.1],
        'n_estimators': [100],
    },
    'rf': {
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    },
    'gb': {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    }
}
test_bs_params = {
    'knn': {
        'n_neighbors': Integer(1, 2),
        'p': Real(1, 5),
    },
    'mlp': {
        'hidden_layer_sizes': Integer(2, 3),
        'max_iter': Integer(10, 11),
        'learning_rate_init': Real(10 ** (-6), 10 ** (-5)),
        'activation': Categorical(['relu']),
        'solver': Categorical(['adam']),
    },
    'ab': {
        'learning_rate': Real(10 ** (-3), 10 ** (-2)),
        'n_estimators': Integer(10, 11),
    },
    'rf': {
        'n_estimators': Integer(10, 11),
        'min_samples_split': Integer(2, 3),
        'min_samples_leaf': Integer(1, 2),
    },
    'gb': {
        'learning_rate': Real(10 ** (-3), 10 ** (-2)),
        'n_estimators': Integer(10, 11),
        'min_samples_split': Integer(2, 3),
        'min_samples_leaf': Integer(1, 2),
    },
}
test_params = {
    'Grid': test_gs_params,
    'Bayes': test_bs_params,
}

default_classifiers = {
    'knn': KNeighborsClassifier,
    'mlp': MLPClassifier,
    'ab': AdaBoostClassifier,
    'rf': RandomForestClassifier,
    'gb': GradientBoostingClassifier,
}
default_regressors = {
    'knn': KNeighborsRegressor,
    'mlp': MLPRegressor,
    'ab': AdaBoostRegressor,
    'rf': RandomForestRegressor,
    'gb': GradientBoostingRegressor,
}
default_models = {
    'Classification': default_classifiers,
    'Regression': default_regressors,
}


# StratifiedKFold doesn't support one-hot encoding, while roc_auc requires it
def custom_roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr'):
    y_true = label_binarize(y_true, classes=list(range(6)))
    y_pred = label_binarize(y_pred, classes=list(range(6)))
    return roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class)


classification_scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='micro'),
    'precision_score': make_scorer(precision_score, average='micro'),
    'recall_score': make_scorer(recall_score, average='micro'),
    'roc_auc_score': make_scorer(custom_roc_auc_score),
}
regression_scorers = {
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error),
    'max_error': make_scorer(max_error),
    'R2': make_scorer(r2_score),
}
default_scorers = {
    'Classification': classification_scorers,
    'Regression': regression_scorers,
}
