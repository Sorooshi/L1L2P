import keras_tuner as kt
import numpy as np

from common.metrics import get_tune_metrics
from models.models import build_model
from common.utils import get_cv

MAX_EPOCHS = 1000
NUM_TUNING_CV_SPLITS = 5

def suggest_hp_base_cnn(hp: kt):
    hp.Choice(name='activation', values=['tanh', 'relu', 'softmax'])
    hp.Int(name='pool_dim', min_value=2, max_value=16)
    hp.Int(name='conv1_filter', min_value=16, max_value=128, step=8)
    hp.Int(name='conv1_kernel', min_value=1, max_value=25, step=2)
    hp.Int(name='conv2_filter', min_value=16, max_value=128, step=8)
    hp.Int(name='conv2_kernel', min_value=1, max_value=25, step=2)
    hp.Int(name='conv3_filter', min_value=16, max_value=128, step=8)
    hp.Int(name='conv3_kernel', min_value=1, max_value=25, step=2)
    hp.Float(name='dropout', min_value=0.1, max_value=0.8)
    hp.Int(name='dense', min_value=2, max_value=200)
    hp.Float(name='lr', min_value=10 ** (-6), max_value=10 ** (-2))
    hp.Choice(name='optimizer', values=['adam', 'sgd', 'nadam'])
    hp.Int(name='epochs', min_value=2, max_value=MAX_EPOCHS)


def suggest_hp_multiinput_cnn(hp: kt):
    hp.Choice(name='activation', values=['tanh', 'relu', 'softmax'])
    hp.Int(name='pool_dim', min_value=2, max_value=16)
    hp.Int(name='conv1_filter', min_value=16, max_value=128, step=8)
    hp.Int(name='conv1_kernel', min_value=1, max_value=25, step=2)
    hp.Int(name='conv2_filter', min_value=16, max_value=128, step=8)
    hp.Int(name='conv2_kernel', min_value=1, max_value=25, step=2)
    hp.Int(name='conv3_filter', min_value=16, max_value=128, step=8)
    hp.Int(name='conv3_kernel', min_value=1, max_value=25, step=2)
    hp.Float(name='dropout', min_value=0.1, max_value=0.8)
    hp.Int(name='dense_img', min_value=32, max_value=256)
    hp.Int(name='dense_mlp', min_value=16, max_value=64)
    hp.Int(name='dense_concat', min_value=16, max_value=128)
    hp.Float(name='lr', min_value=10 ** (-6), max_value=10 ** (-2))
    hp.Choice(name='optimizer', values=['adam', 'sgd', 'nadam'])
    hp.Int(name='epochs', min_value=2, max_value=MAX_EPOCHS)


suggest_hp = {
    'base_cnn': suggest_hp_base_cnn,
    'multiinput_cnn': suggest_hp_multiinput_cnn
}


class CVTuner(kt.engine.tuner.Tuner):
    def __init__(self, demo, model, method, set_params, objective, max_trials, project_name, batch_size):
        oracle = kt.oracles.BayesianOptimizationOracle(objective=kt.Objective(objective, direction='min'), max_trials=max_trials)
        super(CVTuner, self).__init__(hypermodel=suggest_hp[model], oracle=oracle, project_name=project_name)

        self.demo = demo
        self.model = model
        self.method = method
        self.objective = objective
        self.batch_size = batch_size
        self.set_params = set_params
        self.losses = []

    def run_trial(self, trial, X, y):
        def iterate_cv(train_idx, test_idx):
            train_data = {'img': X[train_idx]}
            test_data = {'img': X[test_idx]}
            if 'multiinput' in self.model:
                train_data['demo'] = train_demo[train_idx]
                test_data['demo'] = train_demo[test_idx]
            model_instance = build_model[self.model][self.method](**trial.hyperparameters.values)
            model_instance.fit(train_data, y[train_idx], self.batch_size, epochs, verbose=0)
            y_pred = model_instance.predict(test_data)
            return get_tune_metrics[self.objective](y[test_idx], y_pred)
            # return model_instance.evaluate(test_data, y[test_idx], verbose=0)

        cv = get_cv[self.method](n_splits=NUM_TUNING_CV_SPLITS)
        train_demo = np.hstack((self.demo[:, 3:], np.reshape(1 - self.demo[:, -1], (self.demo.shape[0], 1)))).astype('float32')
        self.hypermodel.build(trial.hyperparameters)
        for key, value in self.set_params.items():
            trial.hyperparameters.values[key] = value
        epochs = trial.hyperparameters.values['epochs']
        val_loss = np.mean([iterate_cv(*idx) for idx in cv.split(X, y, self.demo[:, 0])])
        self.losses.append(val_loss)
        self.oracle.update_trial(trial.trial_id, {self.objective: val_loss})
