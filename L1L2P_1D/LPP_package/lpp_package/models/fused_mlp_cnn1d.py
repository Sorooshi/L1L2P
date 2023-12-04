import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
import lpp_package.common.utils as util


tfk = tf.keras
tfkl = tf.keras.layers
BATCH_SIZE = 6


class FusedCnnMlpMlp(tfk.Model):
    def __init__(self, n_demo_features: int = 1, n_fix_features: int = 1, n_units: int = 10, n_steps: int = 5,
                 target_is_language: int = 1, n_outputs: int = 1, n_filters: int = 32, kernel_size: int = 4, ):
        super().__init__()
        self.conv1 = tfkl.Conv1D(filters=n_filters, kernel_size=kernel_size, activation="tanh",
                                 padding="causal", input_shape=(n_steps, n_fix_features,),
                                 name="conv1")
        self.drop1 = tfkl.Dropout(rate=0.1, name="drop1")
        self.bn1 = tfkl.BatchNormalization(name="bn1")

        self.conv2 = tfkl.Conv1D(filters=n_filters, kernel_size=int(2 * kernel_size - 2),
                                 activation="relu", padding="causal",
                                 name="conv1")
        self.drop2 = tfkl.Dropout(rate=0.1, name="drop2")
        self.mp2 = tfkl.MaxPooling1D(name="mp2")

        self.conv3 = tfkl.Conv1D(filters=n_filters, kernel_size=int(2 * kernel_size - 1),
                                 activation="relu", padding="causal", name="conv3")
        self.drop3 = tfkl.Dropout(rate=0.1, name="drop3")
        self.mp3 = tfkl.MaxPooling1D(name="mp3")

        self.fix_flat = tfkl.Flatten(name="flat")
        self.fix_dense1 = tfkl.Dense(units=32, activation="relu", name="fix_dense1")
        self.fix_dense2 = tfkl.Dense(units=16, activation="relu", name="fix_dense2")

        self.dense1_demo = tfkl.Dense(units=int(2 * n_units), activation="relu",
                                      input_shape=(n_demo_features,), name="demo_dense1")
        self.dense2_demo = tfkl.Dense(units=n_units, name="demo_dense2")

        self.combined_dense1 = tfkl.Dense(units=int(2 * n_units), name="pred_dense1")
        self.combined_bn1 = tfkl.BatchNormalization(name="combined_bn1")
        self.combined_dense2 = tfkl.Dense(units=n_units, name="pred_dense1")

        if target_is_language == 0:  # predicting L2 average scores (regression)
            self.combined_pred = tf.keras.layers.Dense(units=n_outputs, activation="linear", name="predictions")
        elif target_is_language == 1:  # predicting L1 classes
            self.combined_pred = tf.keras.layers.Dense(units=n_outputs, activation="softmax", name="predictions")
        else:
            assert False, "ill-defined target values"

    def call(self, inputs, training=None, mask=None):
        fix_input, demo_input = inputs[0], inputs[1]

        # CNN for Fix.
        a = self.conv1(fix_input)  # training=training
        a = self.drop1(a)
        a = self.bn1(a)
        a = self.conv2(a)
        a = self.drop2(a)
        a = self.mp2(a)
        a = self.conv3(a)
        a = self.drop3(a)
        a = self.mp3(a)
        a = self.fix_flat(a)
        a = self.fix_dense1(a)
        a = self.fix_dense2(a)

        # MLP for Demo.
        b = self.dense1_demo(demo_input)
        b = self.dense2_demo(b)

        # MLP for  combined Fix. + Demo.
        combined = tfkl.concatenate([a, b])
        c = self.combined_dense1(combined)
        c = self.combined_bn1(c)
        c = self.combined_dense2(c)
        return self.combined_pred(c)


class TrainTestFusedModels:
    def __init__(self, configs, data: dict = None, n_steps: int = 15,
                 n_epochs: int = 1, target_is_language: int = 1, ):
        super().__init__()

        self.data = data
        self.configs = configs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.target_is_language = target_is_language
        self.params = defaultdict()
        self.results = defaultdict(defaultdict)

        # if self.target_is_language != 1:  # predicting L2 average scores (regression)
        #     # self.metrics_fn = ["mean_absolute_error"]
        #     # self.loss_fn = tfk.losses.MeanAbsolutePercentageError()
        #     self.activation = None
        # else:  # predicting L1 classes (always multiclass)
        #     # self.metrics_fn = ["accuracy"]
        #     # self.loss_fn = tfk.losses.SparseCategoricalCrossentropy()
        #     self.activation = "softmax"

    @staticmethod
    def splitting_window(features, targets, n_steps):
        x_seq, y_seq = [], []
        for i in range(len(features)):
            end_idx = i + n_steps
            if end_idx >= len(features):
                break
            x_seq.append(features[i:end_idx, :])
            y_seq.append(targets[end_idx])

        return np.asarray(x_seq), np.asarray(y_seq)

    def train_test_cnn1dmlps_estimator(self, ):
        """ returns of dict of dicts, containing y_test and y_pred per each repeat. """
        for k, v in self.data.items():
            if int(k) >= 6:
                break
            self.results[k] = defaultdict()
            x_train = v['x_train']
            y_train = v['y_train']
            x_test = v['x_test']
            y_test = v['y_test']

            if self.target_is_language == 1:
                y_train = np.subtract(y_train, 1, dtype=np.int8)  # [:x_train.shape[0]]
                y_test = np.subtract(y_test, 1, dtype=np.int8)  # [:x_test.shape[0]]

            # creating windowed fixation data
            x_train_fix_windowed, y_train_fix_windowed = self.splitting_window(
                x_train[:, :3], y_train, self.n_steps)

            x_test_fix_windowed, y_test_fix_windowed = self.splitting_window(
                x_test[:, :3], y_test, self.n_steps)

            x_train_demo = x_train[:len(y_train_fix_windowed), 3:]
            x_test_demo = x_train[:len(y_test_fix_windowed), 3:]
            y_train_demo = y_train[:len(y_train_fix_windowed)]
            y_test_demo = y_test[:len(y_test_fix_windowed)]

            # predicting L2 average scores (regression)
            if self.target_is_language == 0:
                metrics = None  # ["mean_absolute_error"]
                n_outputs = 1
                loss_fn = tfk.losses.MeanAbsolutePercentageError()
            # predicting L1 classes
            elif self.target_is_language == 1:
                metrics = None  # ["accuracy", "accuracy", "accuracy"]
                # metrics = tf.metrics.Metric.Acc
                loss_fn = tfk.losses.SparseCategoricalCrossentropy()
                n_outputs = len(np.unique(v['y_train']))
            else:
                print("ill-defined target_is_value argument")

            print(
                f" Predicting L1 or L2: {self.target_is_language}"
                f" Number of classes: {n_outputs} \n"
                f" x_train (fix, demo): {x_train_fix_windowed.shape, x_train_demo.shape} \n"
                f" y_train (fix, demo): {y_train_fix_windowed.shape, y_train_demo.shape} \n"
                f" x_test (fix, demo): {x_test_fix_windowed.shape, x_test_demo.shape} \n"
                f" y_test (fix, demo): {y_test_fix_windowed.shape, y_test_demo.shape} \n"
                f"y_true_demo: {y_test_demo} \n"
                f"y_true_fix: {y_test_fix_windowed} \n"
            )

            model = FusedCnnMlpMlp(
                n_demo_features=x_train_demo.shape[1], n_fix_features=x_train_fix_windowed.shape[1],
                n_units=32, n_steps=self.n_steps, target_is_language=self.target_is_language,
                n_outputs=n_outputs, n_filters=64, kernel_size=3,
            )

            optimizer = tfk.optimizers.Adam(learning_rate=1e-3)

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics,
            )

            start = time.time()
            history = model.fit(
                x=[x_train_fix_windowed, x_train_demo],
                y=[y_train_fix_windowed, y_train_demo],
                batch_size=BATCH_SIZE, epochs=self.n_epochs, verbose=2,
            )

            print(
                "summary of the model: \n ",
                model.summary()
            )

            if self.target_is_language == 1:
                y_pred_prob = model.predict([x_test_fix_windowed, x_test_demo])
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.target_is_language == 0:
                y_pred = model.predict([x_test_fix_windowed, x_test_demo])
                y_pred_prob = None

            if y_pred_prob is not None:
                print(
                    f" y_pred_prob: {y_pred_prob.shape} \n"
                    f" y_pred: {y_pred.shape} \n"
                    f" y_pred squeezed: {np.squeeze(y_pred).shape} \n"
                    f" y_true: {y_test_demo.shape} \n"
                )
            else:
                if y_pred_prob is not None:
                    print(
                        f" y_pred: {y_pred.shape} \n"
                        f" y_true: {y_test_demo.shape} \n"
                    )
            end = time.time()
            self.results[k]["y_test"] = y_test_demo
            self.results[k]["x_test"] = x_test_demo
            self.results[k]["y_pred"] = y_pred
            self.results[k]["y_pred_prob"] = y_pred_prob
            self.results[k]["history"] = history
            self.results[k]["exe_time"] = end - start

        # save results
        util.save_a_dict(
            a_dict=self.results,
            name=self.configs.specifier,
            save_path=self.configs.results_path,
        )

    def load_saved_tuned_params(self, ):
        saved_tuned_params = util.load_a_dict(
            name=self.configs.specifier,
            save_path=self.configs.params_path
        )
        return saved_tuned_params

    def print_results(self, ):
        results = util.load_a_dict(
            name=self.configs.specifier,
            save_path=self.configs.results_path,
        )
        util.print_the_evaluated_results(
            results,
            self.configs.learning_method,
        )


