from sklearn.model_selection import train_test_split
import argparse
import datetime
import pickle
import json
import time

from common.utils import parse_test_description, make_msg, pickle_dict, get_gpu
from models.models import allowed_models, allowed_methods
from common.metrics import get_method_metrics
from data.heatdata import make_img_dataset
from models.img_search import kt_search

set_params_ = 'set_params.json'
max_trials_ = 50

CLI = argparse.ArgumentParser(prog='HyperOpt+CV')
CLI.add_argument('-d', '--test_desc', type=str, required=True)
CLI.add_argument('-m', '--model', type=str, required=True)
CLI.add_argument('-p', '--set_params', type=str)
CLI.add_argument('-o', '--tuning_objective', type=str)
CLI.add_argument('-t', '--max_trials', type=int)
CLI.add_argument('--debug', action='store_true')

if __name__ == '__main__':
    args = CLI.parse_args()
    test_desc_ = args.test_desc
    if args.model is not None:
        model_ = args.model
    if args.max_trials is not None:
        max_trials_ = args.max_trials
    if args.tuning_objective is not None:
        objective = args.tuning_objective
    else:
        objective = get_method_metrics

    date = datetime.datetime.now().strftime("%H:%M-%d-%m")
    name_ = f'{test_desc_}-{model_}-{date}'
    run_params = parse_test_description(test_desc_)
    model, method = model_.split('-')
    assert model in allowed_models, f'{model} model isn\'t implemented'
    assert method in allowed_methods, f'{method} doesn\'t exist'

    with open(f'Pickle/params/{set_params_}') as f:
        if set_params_.endswith('.pickle'):
            set_params = pickle.load(f)
        elif set_params_.endswith('.json'):
            set_params = json.load(f)

    get_gpu()
    X, y, demo = make_img_dataset(**run_params, method=method)
    if args.debug:
        stratify = y if method == 'clf' else None
        X, _, y, _, demo, _ = train_test_split(X, y, demo, stratify=y, train_size=250)
        max_trials_ = 1

    st = time.time()
    params, scores, predictions, history, tune_history = kt_search(X, y, demo, model, method, set_params, objective,
                                                                   max_trials_, project_name=f'kt/{name_}')
    et = time.time()
    elapsed_time = str(datetime.timedelta(seconds=int(et-st)))

    msg = make_msg(name_, params, scores, elapsed_time)
    print(msg)
    pickle_dict(name_, params, save_path='Pickle/params')
    pickle_dict(name_, scores, save_path='Pickle/results')
    pickle_dict(name_, predictions, save_path='Pickle/predictions')
    pickle_dict(name_, history, 'Pickle/history')
    pickle_dict(name_, tune_history, 'Pickle/tune_history')
