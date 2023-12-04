from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    r2_score, mean_absolute_percentage_error, mean_absolute_error
from keras.losses import sparse_categorical_crossentropy

from sklearn.preprocessing import OneHotEncoder
import keras.backend as K
import numpy as np


def multiclass_roc_auc(y_true, y_pred, average):
    ohe = OneHotEncoder(sparse_output=False)
    y_true_ohe = ohe.fit_transform(y_true.reshape(-1, 1))
    return roc_auc_score(y_true_ohe, y_pred, average=average, multi_class='ovr')


def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> dict[str, float]:
    y_pred_max = np.argmax(y_pred, axis=-1)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_max),
        'precision': precision_score(y_true, y_pred_max, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred_max, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred_max, average=average, zero_division=0),
        'roc_auc': multiclass_roc_auc(y_true, y_pred, average)
    }
    return metrics


def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics


get_metrics = {
    'clf': get_classification_metrics,
    'reg': get_regression_metrics
}


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


get_method_metrics = {
    'clf': 'categorical_crossentropy',
    'reg': 'rmse'
}
get_tune_metrics = {
    'categorical_crossentropy': sparse_categorical_crossentropy,
    'rmse': root_mean_squared_error,
    'roc_auc': multiclass_roc_auc,
}