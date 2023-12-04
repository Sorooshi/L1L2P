import argparse
import os
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from lpp_package.models.baseline import BaseLineModel
from lpp_package.data.preprocess import preprocess_data
from lpp_package.data.language_data import LanguageData

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

np.set_printoptions(suppress=True, precision=3, linewidth=140)


def args_parser(arguments):
    _pp = arguments.pp.lower()
    _run = arguments.run
    _data_name = arguments.data_name.lower()
    _estimator_name = arguments.estimator_name.lower()
    _project = arguments.project
    _target_is_org = arguments.target_is_org
    _to_shuffle = arguments.to_shuffle
    _n_clusters = arguments.n_clusters
    _target_is_language = arguments.target_is_language
    _n_epochs = arguments.n_epochs
    _n_filters = arguments.n_filters

    return (_pp, _run, _data_name, _estimator_name, _project,
            _to_shuffle, _n_clusters, _target_is_language, _n_epochs, _n_filters, _target_is_org)


configs = {
    "models_path": Path("/home/sshalileh/LPP/Models"),
    "results_path": Path("/home/sshalileh/LPP/Results"),
    "figures_path": Path("/home/sshalileh/LPP/Figures"),
    "params_path": Path("/home/sshalileh/LPP/Params"),
    "n_repeats": 10,
    "n_splits": 5,
}

configs = SimpleNamespace(**configs)

if not configs.models_path.exists():
    configs.models_path.mkdir()

if not configs.results_path.exists():
    configs.results_path.mkdir()

if not configs.figures_path.exists():
    configs.figures_path.mkdir()

if not configs.params_path.exists():
    configs.params_path.mkdir()


if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project", type=str, default="lpp",
        help="Project name (for WandB project initialization)."
    )

    parser.add_argument(
        "--data_name", type=str, default="lpp_Fix_Demo",
        help="Dataset's name, e.g., lpp_Ru_Fix_Demo."
             "The following (lowercase) strings are supported"
             "  1) Demographic + Fixation_report of Russian Native speaker = lpp_ru_fix_demo,"
             "  2) Demographic + Fixation_report of Italian Native speaker = lpp_it_fix_demo,"
             "  3) Demographic + Fixation_report of Finish Native speaker = lpp_fi_fix_demo,"
             "  4) etc "
             "  5) Demographic + Fixation_report of all nations = lpp_all_fix_demo,"  # we should add these cases
    )

    parser.add_argument(
        "--estimator_name", type=str, default="base_reg",
        help="None case sensitive first letter abbreviated name of an estimator proceeds "
             "  one of the three following suffixes separated with the underscore."
             "  Possible suffixes are: regression := reg, "
             "  classification := cls, "
             "  E.g., Random Forest Regressor := rf_reg, or "
             "  Random Forest Classifiers := rf_cls "
             "Note: First letter of the methods' name should be used for abbreviation."
    )

    parser.add_argument(
        "--run", type=int, default=1,
        help="Run the model or load the saved"
             " weights and reproduce the results."
    )

    parser.add_argument(
        "--pp", type=str, default="mm",
        help="Data preprocessing method:"
             " MinMax/Z-Scoring/etc."
    )

    parser.add_argument(
        "--n_clusters", type=int, default=3,
        help="Number of clusters/classes/discrete target values."
    )

    parser.add_argument(
        "--target_is_org", type=int, default=1,
        help="Whether to preprocessed target values (in regression tasks) or not."
    )

    parser.add_argument(
        "--to_shuffle", type=int, default=1,
        help="Whether to shuffle data during CV or not."
             "  Only setting it to one (shuffle=1) will shuffle data."
    )

    parser.add_argument(
        "--target_is_language", type=int, default=0,
        help="Whether to predict L1 )when target_is_language=1) or to predict L2 (target_is_language=0)."
    )

    parser.add_argument(
        "--n_epochs", type=int, default=None,
        help="The number of epochs to train a neural network."
    )

    parser.add_argument(
        "--n_filters", type=int, default=64,
        help="The number of filters in CNN layers."
    )

    args = parser.parse_args()

    (pp, run, data_name, estimator_name, project, to_shuffle,
     n_clusters, target_is_language, n_epochs, n_filters, target_is_org) = args_parser(arguments=args)

    print(
        "configuration: \n",
        "  estimator:", estimator_name, "\n",
        "  data_name:", data_name, "\n",
        "  shuffle_data:", to_shuffle, "\n",
        "  pre-processing:", pp, "\n",
        "  run:", run, "\n",
        "  n_epochs:", n_epochs, "\n",
        "  target_is_language:", target_is_language, "\n",
    )

    lpp = LanguageData(
        n_splits=configs.n_splits,
        n_repeats=configs.n_repeats,
        data_name=data_name,
    )

    # Determine which dataset to use, e.g. demo dataset
    # alone or concatenation of demo and IA_report, for instance.
    if data_name == "lpp_ru_fix_demo":
        # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
        demo = lpp.get_ru_fix_demo_datasets()  # demos and fixation

        df_data_to_use = demo.loc[:, ['Fix_X', 'Fix_Y', 'Fix_Duration', 'motiv', 'IQ',
                                      'Age', 'Sex', 'Target_Ave', 'Target_Label']]
        c_features = ['Sex']
        indicators = ['SubjectID']
        targets = ['Target_Ave', 'Target_Label']

    elif data_name == "lpp_all_fix_demo":
        if target_is_language == 0:
            # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
            demo = lpp.get_all_fix_demo_datasets()  # demos and fixation
            df_data_to_use = demo.loc[:, ['Fix_X', 'Fix_Y', 'Fix_Duration', 'motiv', 'IQ',
                                          'Age', 'Sex', 'Target_Ave', 'Target_Label']]
            c_features = ['Sex']
            indicators = ['SubjectID']
            targets = ['Target_Ave', 'Target_Label']

        # Predicting native Language (country of origin, L1)
        elif target_is_language == 1:
            # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
            demo = lpp.get_all_fix_demo_language_datasets()  # demos and fixation
            df_data_to_use = demo.loc[:, ['Fix_X', 'Fix_Y', 'Fix_Duration', 'motiv', 'IQ',
                                          'Age', 'Sex', 'Language']]
            c_features = ['Sex']
            indicators = ['SubjectID']
            targets = ['Language']

    elif data_name == "lpp_ru_fix":
        # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
        demo = lpp.get_ru_fix_demo_datasets()  # demos and fixation
        df_data_to_use = demo.loc[:, ['Fix_X', 'Fix_Y', 'Fix_Duration',
                                      'Target_Ave', 'Target_Label']] # fixation data only
        c_features = []
        indicators = ['SubjectID']
        targets = ['Target_Ave', 'Target_Label']

    elif data_name == "lpp_all_fix":
        if target_is_language == 0:
            # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
            demo = lpp.get_all_fix_demo_datasets()  # demos and fixation
            df_data_to_use = demo.loc[:, ['Fix_X', 'Fix_Y', 'Fix_Duration',
                                          'Target_Ave', 'Target_Label']]  # fixation data only
            c_features = []
            indicators = ['SubjectID']
            targets = ['Target_Ave', 'Target_Label']
        if target_is_language == 1:
            demo = lpp.get_all_fix_demo_language_datasets()  # demos and fixation
            df_data_to_use = demo.loc[:, ['Fix_X', 'Fix_Y', 'Fix_Duration', 'Language']]  # fixation data only
            c_features = []
            indicators = ['SubjectID']
            targets = ['Language']

    elif data_name == "lpp_ru_demo":
        # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
        demo = lpp.get_ru_demo_datasets()  # demos and fixation
        df_data_to_use = demo.loc[:, ['motiv', 'IQ',
                                      'Age', 'Sex', 'Target_Ave', 'Target_Label']]
        c_features = ['Sex']
        indicators = ['SubjectID']
        targets = ['Target_Ave', 'Target_Label']

    elif data_name == "lpp_all_demo":
        # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
        demo = lpp.get_all_demo_datasets()  # demos and fixation
        df_data_to_use = demo.loc[:, ['motiv', 'IQ',
                                      'Age', 'Sex', 'Target_Ave', 'Target_Label']]
        c_features = ['Sex']
        indicators = ['SubjectID']
        targets = ['Target_Ave', 'Target_Label']

    else:
        print("data_name argument:", data_name)
        assert False, "Ill-defined data_name argument. Refer to help of data_name argument for more."

    x_org, y_org = lpp.get_onehot_features_targets(
        data=df_data_to_use,
        c_features=c_features,
        indicators=indicators,
        targets=targets,
    )

    if data_name == "lpp_all_fix_demo":
        features_fixed_order = ['Fix_X', 'Fix_Y', 'Fix_Duration', 'motiv', 'IQ', 'Age',
                                'Sex_0', 'Sex_1', ]
        x_org = x_org[features_fixed_order]

    if estimator_name.split("_")[-1] == "reg":
        learning_method = "regression"
        y = y_org.Target_Ave.values
        from lpp_package.models.regression_estimators import RegressionEstimators
        from lpp_package.models.cnn1d_estimator import TrainTest1dCNN
        from lpp_package.models.fused_mlp_cnn1d import TrainTestFusedModels

    elif estimator_name.split("_")[-1] == "cls":
        learning_method = "classification"
        y = y_org.Language.values
        from lpp_package.models.classification_estimators import ClassificationEstimators
        from lpp_package.models.cnn1d_estimator import TrainTest1dCNN
        from lpp_package.models.fused_mlp_cnn1d import TrainTestFusedModels

    elif estimator_name.split("_")[-1] == "clu":
        learning_method = "clustering"
        print("No supported")
        assert False, "unsupported method"

    elif estimator_name.split("_")[-1] == "ad":
        learning_method = "abnormality_detection"
        print("No supported")
        assert False, "unsupported method"

    else:
        assert False, "Undefined algorithm and thus undefined target values"

    if to_shuffle == 1:
        to_shuffle = True
        group = learning_method + "-" + "shuffled"
    else:
        to_shuffle = False
        group = learning_method + "-" + "not-shuffled"

    if target_is_org == 1:
        target_is_org = True
    else:
        target_is_org = False

    # Adding some details for the sake of clarity in storing and visualization
    configs.run = run
    configs.project = project
    configs.group = group

    if n_epochs is not None:
        specifier = data_name + "-" + estimator_name + \
                    "--shuffled:" + str(to_shuffle) + \
                    "--target is language:" + str(target_is_language) + \
                    "--n_epochs: " + str(n_epochs)
    else:
        specifier = data_name + "-" + estimator_name + \
                    "--shuffled:" + str(to_shuffle) + \
                    "--target is language:" + str(target_is_language)

    configs.specifier = specifier
    configs.data_name = data_name
    configs.name_wb = data_name + ": " + specifier
    configs.learning_method = learning_method
    configs.n_clusters = n_clusters

    x = preprocess_data(x=x_org, pp=pp)  # only x is standardized
    if target_is_org is False:
        y = preprocess_data(x=y, pp=pp)

    cv = lpp.get_stratified_kfold_cv(
        to_shuffle=to_shuffle,
        n_splits=configs.n_splits
    )

    if target_is_language == 1:
        data = lpp.get_stratified_train_test_splits(
            x=x, y=y,
            labels=y_org.Language.values,
            to_shuffle=to_shuffle,
            n_splits=configs.n_repeats
        )

    elif target_is_language == 0:
        data = lpp.get_stratified_train_test_splits(
            x=x, y=y,
            labels=y_org.Target_Label.values,
            to_shuffle=to_shuffle,
            n_splits=configs.n_repeats
        )

    print(
        "x_org:", x_org.shape, "\n", x_org.head(), "\n",
        "y_org:", y_org.shape, "\n", y_org, "\n",
        "y:", y.shape, "\n", y,
    )

    # Baseline models (random prediction)
    if estimator_name == "base_reg" or \
            estimator_name == "base_cls" or \
            estimator_name == "base_clu":
        blm = BaseLineModel(
            y_train=y,
            learning_method=learning_method,
            configs=configs,
            test_size=1000
        )

        blm.repeat_random_pred()
        blm.save_results()
        blm.print_results()

        assert False, "Random prediction is done, no need to proceed further"

    # Regression methods: tune and fit
    if learning_method == "regression" and run == 1:

        reg_est = RegressionEstimators(
            x=x, y=y, cv=cv, data=data,
            estimator_name=estimator_name,
            configs=configs,
        )

        reg_est.instantiate_tuning_estimator_and_parameters()
        reg_est.tune_hyper_parameters()
        reg_est.instantiate_train_test_estimator()
        reg_est.train_test_tuned_estimator()
        reg_est.save_params_results()
        reg_est.print_results()
    # Regression methods: fit with tuned params
    elif learning_method == "regression" and run == 2:

        if "cnn" not in estimator_name.split("_")[0]:
            reg_est = RegressionEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )
            reg_est.instantiate_train_test_estimator()
            reg_est.train_test_tuned_estimator()
            reg_est.save_params_results()

        elif estimator_name.split("_")[0] == "cnn1d":
            reg_est = TrainTest1dCNN(
                data=data, configs=configs,
                n_steps=20, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
            reg_est.train_test_1dcnn_estimator()

        elif estimator_name.split("_")[0] == "cnn1dmlps":
            reg_est = TrainTestFusedModels(
                data=data, configs=configs,
                n_steps=20, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
            reg_est.train_test_cnn1dmlps_estimator()

        elif estimator_name.split("_")[0] == "cnn2d":
            print("modify it later")
        elif estimator_name.split("_")[0] == "cnn_lstm2d":
            print("modify it later")

        reg_est.print_results()
    # Regression methods: print the saved results
    elif learning_method == "regression" and run == 3:
        if "cnn" not in estimator_name.split("_")[0]:
            reg_est = RegressionEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

        elif estimator_name.split("_")[0] == "cnn1d":
            reg_est = TrainTest1dCNN(
                data=None, configs=configs,
                n_steps=10, n_epochs=n_epochs,
                target_is_language=target_is_language
            )

        elif estimator_name.split("_")[0] == "cnn1dmlps":
            reg_est = TrainTestFusedModels(
                data=data, configs=configs,
                n_steps=20, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
        reg_est.print_results()

    # Classification methods: tune and fit
    elif learning_method == "classification" and run == 1:

        cls_est = ClassificationEstimators(
            x=x, y=y, cv=cv, data=data,
            estimator_name=estimator_name,
            configs=configs,
        )
        cls_est.instantiate_tuning_estimator_and_parameters()
        cls_est.tune_hyper_parameters()
        cls_est.instantiate_train_test_estimator()
        cls_est.train_test_tuned_estimator()
        cls_est.save_params_results()
        cls_est.print_results()

    # Classification methods: fit with tuned params
    elif learning_method == "classification" and run == 2:
        if "cnn" not in estimator_name.split("_")[0]:
            cls_est = ClassificationEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )
            cls_est.instantiate_train_test_estimator()
            cls_est.train_test_tuned_estimator()
            cls_est.save_params_results()

        elif estimator_name.split("_")[0] == "cnn1d":
            cls_est = TrainTest1dCNN(
                data=data, configs=configs,
                n_steps=20, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
            cls_est.train_test_1dcnn_estimator()

        elif estimator_name.split("_")[0] == "cnn1dmlps":
            cls_est = TrainTestFusedModels(
                data=data, configs=configs,
                n_steps=20, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
            cls_est.train_test_cnn1dmlps_estimator()

        elif estimator_name.split("_")[0] == "cnn2d":
            print("modify it later")
        elif estimator_name.split("_")[0] == "cnn_lstm2d":
            print("modify it later")
        cls_est.print_results()
    # Classification methods: print the saved results
    elif learning_method == "classification" and run == 3:
        if "cnn" not in estimator_name.split("_")[0]:
            cls_est = ClassificationEstimators(
                x=None, y=None, cv=None, data=None,
                estimator_name=estimator_name,
                configs=configs,
            )
        elif estimator_name.split("_")[0] == "cnn1d":
            cls_est = TrainTest1dCNN(
                data=None, configs=configs,
                n_steps=10, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
        elif estimator_name.split("_")[0] == "cnn1dmlps":
            cls_est = TrainTestFusedModels(
                data=data, configs=configs,
                n_steps=20, n_epochs=n_epochs,
                target_is_language=target_is_language
            )
        cls_est.print_results()

    elif run == 1:
        print(
            "\n Hyper-parameters tuning and train-test evaluation at " + data_name + " are finished. \n",
            "  The corresponding results, parameters, models, and figures of " + estimator_name + " are stored."
        )

    elif run == 2:
        print(
            "\n Train-test evaluation at " + data_name + " are finished. \n",
            "  The corresponding results, parameters, models, and figures of " + estimator_name + " are stored."
        )
    print(
        "configuration: \n",
        "  estimator:", estimator_name, "\n",
        "  data_name:", data_name, "\n",
        "  shuffle_data:", to_shuffle, "\n",
        "  pre-processing:", pp, "\n",
        "  run:", run, "\n",
        "  target_is_language:", target_is_language, "\n",
    )
