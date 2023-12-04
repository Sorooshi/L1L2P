from l2.common.utils import parse_test_description, make_msg, send_tg_message
from l2.models.search_methods import search_cv
from l2.data.split_data import make_dataset
import argparse
import time

lang_ = model_ = None
input_ = 'Data/gauss/'
n_jobs_ = 4
method_ = 'Classification'
target_ = 'Target_Label'
search_ = 'Grid'


CLI = argparse.ArgumentParser(prog='GridSearch+CV results')
CLI.add_argument('-l', '--lang', type=str, required=True)
CLI.add_argument('-m', '--model',  type=str, required=True)
CLI.add_argument('--method', type=str)
CLI.add_argument('--target', type=str)
CLI.add_argument('--search', type=str)
CLI.add_argument('-i', '--input', type=str)
CLI.add_argument('-j', '--jobs', type=int)
CLI.add_argument('-d', '--debug', action='store_true')
CLI.add_argument('-s', '--send_to_tg', action='store_true')

if __name__ == '__main__':
    args = CLI.parse_args()
    if args.lang is not None:
        lang_ = args.lang
    if args.model is not None:
        model_ = args.model
    if args.method is not None:
        method_ = args.method
    if args.target is not None:
        target_ = args.target
    if args.search is not None:
        search_ = args.search
    if args.input is not None:
        input_ = args.input
    if args.jobs is not None:
        n_jobs_ = args.jobs
    debug = args.debug
    send_to_tg = args.send_to_tg

    cur_params = parse_test_description(lang_)
    X, y, value_idx = make_dataset(**cur_params, target=target_, path_to_data=input_)

    st = time.time()
    search_params, cv_score = search_cv(X, y, model_, method_, search_, value_idx, n_jobs=n_jobs_, debug=debug)
    et = time.time()

    message = make_msg(model_, method_, lang_, search_params, cv_score, et - st)
    print(message)
    if send_to_tg:
        send_tg_message(message)
