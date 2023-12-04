from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from keras.optimizers import Adam, SGD, Nadam
from keras.optimizers import legacy
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib import ticker
from sys import platform
from typing import Any
import numpy as np
import itertools
import requests
import pickle
import os

from data.split_data import lang_list

get_cv = {
    'clf': StratifiedGroupKFold,
    'reg': GroupKFold
}


def parse_test_description(test_description: str) -> dict[str, Any]:
    """
    Used for command line inputs.
    
    Test description functions as follows:
    lang_1+...+lang_n-data_type-param_1=x+...+param_n=x
    lang_k is a 2-letter language code (such as ru, fi etc.), or write 'all' to get all available languages;
    data_type describes the kind of data type you want to use
    (fix or fix+demo for row-wise and time-series methods, and gauss or scatter for image methods);
    param_k are additional params specific to the chosen data type.

    :param: test_description
    :return: parameters in a dictionary
    """

    test_parameters = {}
    specs = test_description.split('-')
    cur_langs = specs.pop(0).split('+')
    if cur_langs == ['all']:
        cur_langs = lang_list
    test_parameters['lang_names'] = cur_langs
    test_parameters['data_type'] = specs.pop(0)
    spec_dict = {}
    for spec in specs:
        col_name, col_values = spec.split('=')
        spec_dict[col_name] = col_values.split('+')
    test_parameters['params'] = spec_dict
    return test_parameters


def make_msg(test_name, hyperopt_results, cv_results, elapsed_time):
    msg = f'{test_name}.pickle\n\n'
    for name, value in hyperopt_results.items():
        msg += f'{name} = {value}\n'
    msg += '\n'
    for name, (mean, std) in cv_results.items():
        msg += f'{name}: ({mean:.5f}, {std:.5f})\n'
    msg += f'\ntime elapsed: {elapsed_time}\n'
    return msg


def send_tg_message(msg: str) -> None:
    """
    Send a message with a Telegram bot.

    :param msg: message string
    """
    BOT_TOKEN = os.environ.get('TG_BOT_TOKEN')
    CHAT_ID = os.environ.get('TG_CHAT_ID')

    URL = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    formatted_text = f'```\n{msg}\n```'
    msg_data = {
        'chat_id': CHAT_ID,
        'text': formatted_text,
        'parse_mode': 'markdown'
    }
    requests.post(url=URL, json=msg_data)


def pickle_dict(name: str, d: dict[str, Any], save_path: str) -> None:
    """
    Serializes and saves a dictionary with results of tune_img_data.py

    :param d: the dictionary to save
    :param test_description: data used in the test
    :param method: classification or regression
    :param model: model used in the test run
    :param save_path: where to save serialized data
    """
    with open(f'{save_path}/{name}.pickle', 'wb') as file:
        pickle.dump(d, file)


def get_optimizers_for_system():
    if platform == 'darwin':
        opt_dict = {
            'adam': legacy.Adam,
            'sgd': legacy.SGD,
            'nadam': legacy.Nadam,
        }
    else:
        opt_dict = {
            'adam': Adam,
            'sgd': SGD,
            'nadam': Nadam,
        }
    return opt_dict


def get_img_with(X, y, demo, subj=None, not_subj=None, text=None, not_text=None, target=None):
    for i in range(X.shape[0]):
        if ((subj and demo[i]['SubjectID'] == subj) or (not_subj and demo[i]['SubjectID'] != not_subj) or
            (not subj and not not_subj)) and \
           ((text and demo[i]['Text_ID'] == text) or (not_text and demo[i]['Text_ID' != not_text]) or
            (not text and not not_text)) and (target and target == y[i][0] or not target):
            return X[i]


def minmax(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))


def get_dict_of_lists(d):
    res = {}
    for key in d[0].keys():
        scores = [x[key] for x in d]
        res[key] = (np.mean(scores), np.std(scores))
    return res


def plot_history(histories: dict[str, float], method: str = 'clf'):
    if method == 'clf':
        score = 'accuracy'
    else:
        score = 'mrse'
    for h in histories:
        plt.plot(h[score])
    plt.title(f'model {score}')
    plt.ylabel(score)
    plt.xlabel('epoch')
    tick = max(len(histories[0][list(histories[0].keys())[0]]) // 5, 1)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick))
    plt.show()
    # summarize history for loss
    for h in histories:
        plt.plot(h['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def get_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                device=gpus[0],
                logical_devices=[
                    tf.config.LogicalDeviceConfiguration(memory_limit=32000)
                ],
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
