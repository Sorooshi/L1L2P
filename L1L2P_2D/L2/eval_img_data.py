from sklearn.model_selection import train_test_split
import argparse
import datetime
import pickle
import json
import time

from common.utils import parse_test_description, make_msg, pickle_dict, get_gpu
from models.models import allowed_models, allowed_methods
from models.img_search import cross_validate
from data.heatdata import make_img_dataset

params_ = 'params.json'
epochs_ = 25

CLI = argparse.ArgumentParser(prog='CV')
CLI.add_argument('-d', '--test_desc', type=str, required=True)
CLI.add_argument('-m', '--model', type=str, required=True)
CLI.add_argument('-p', '--params', type=str)
CLI.add_argument('-e', '--epochs', type=int)
CLI.add_argument('--debug', action='store_true')

if __name__ == '__main__':
    args = CLI.parse_args()
    test_desc_ = args.test_desc
    model_ = args.model
    if args.params is not None:
        params_ = args.params
    if args.epochs is not None:
        epochs_ = args.epochs

    date = datetime.datetime.now().strftime("%d-%m-%H:%M")
    name_ = f'eval-{test_desc_}-{model_}-{date}'
    run_params = parse_test_description(test_desc_)
    model, method = model_.split('-')
    assert model in allowed_models, f'{model} model isn\'t implemented'
    assert method in allowed_methods, f'{method} doesn\'t exist'

    with open(f'Pickle/params/{params_}') as f:
        if params_.endswith('.pickle'):
            params = pickle.load(f)
        elif params_.endswith('.json'):
            params = json.load(f)
    params['epochs'] = epochs_

    get_gpu()
    X, y, demo = make_img_dataset(**run_params, method=method)
    if args.debug:
        X, _, y, _, demo, _ = train_test_split(X, y, demo, train_size=100)
        params['epochs'] = 1

    st = time.time()
    scores, histories, predictions = cross_validate(X, y, demo, model, method, params)
    et = time.time()
    elapsed_time = str(datetime.timedelta(seconds=int(et - st)))

    msg = make_msg(name_, params, scores, elapsed_time)
    print(msg)
    pickle_dict(name_, scores, save_path='Pickle/results')
    pickle_dict(name_, histories, save_path='Pickle/history')
    pickle_dict(name_, predictions, save_path='Pickle/predictions')
