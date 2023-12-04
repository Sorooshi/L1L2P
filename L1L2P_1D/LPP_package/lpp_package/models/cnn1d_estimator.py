import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
import lpp_package.common.utils as util

tfk = tf.keras
tfkl = tf.keras.layers
BATCH_SIZE = 6


class CNN1D(tf.keras.Model):
    def __init__(self, n_filters: int = 32, n_features: int = 3, kernel_size: int = 3, n_outputs: int = 2,
                 n_steps: int = 10, target_is_language: int = 1, ):  # *args, **kwargs
        # n_filters, n_features, kernel_size, n_outputs, n_steps, target_is_language
        super().__init__()  # *args, **kwargs
        self.n_filters = n_filters
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.target_is_language = target_is_language

        # One can use this formula [(W−K+2P)/S]+1 to compute the output of CNN layer, where;
        #   W is the input data dimension, e.g., 128*128 image;
        #   K is the Kernel size;
        #   P is the padding;
        # self.input_data = tf.keras.layers.InputLayer(shape=(self.n_steps, self.n_features), name="inputs")
        self.conv1 = tf.keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            activation="tanh",
            input_shape=(self.n_steps, self.n_features),
            name="conv1",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.conv2 = tf.keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            activation="relu",
            name="conv2",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.mp2 = tf.keras.layers.MaxPooling1D(name="mp1")
        self.conv3 = tf.keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=int(2 * self.kernel_size - 1),
            activation="relu",
            name="conv3",
        )
        self.drop3 = tf.keras.layers.Dropout(0.1, name="drop1")
        self.conv4 = tf.keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=int(2 * self.kernel_size - 2),
            activation="relu",
            name="conv4",
        )
        self.bn4 = tf.keras.layers.BatchNormalization(name="bn3")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation="tanh")
        self.bn_d1 = tf.keras.layers.BatchNormalization(name="bn4")
        self.dense2 = tf.keras.layers.Dense(units=128, activation="relu")

        # predicting L2 average scores (regression)
        if self.target_is_language != 1:
            self.dense3 = tf.keras.layers.Dense(units=self.n_outputs, activation="linear")

        # predicting L1 classes
        else:
            if self.n_outputs == 1:
                self.dense3 = tf.keras.layers.Dense(units=self.n_outputs, activation="sigmoid")
            else:
                self.dense3 = tf.keras.layers.Dense(units=self.n_outputs, activation="softmax")

    def call(self, inputs, ):  # inputs, training=False, mask=None
        # x = self.input_data(inputs)
        # print("x0:", x.shape)
        x = self.conv1(inputs)  # training=training
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn_d1(x)
        x = self.dense2(x)
        return self.dense3(x)


class TrainTest1dCNN:
    def __init__(self, data, configs, n_steps, n_epochs, target_is_language):
        self.data = data
        self.configs = configs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.params = defaultdict()
        self.results = defaultdict(defaultdict)
        self.target_is_language = target_is_language
        # super().__init__()

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

    def train_test_1dcnn_estimator(self, ):
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

            x_train_windowed, y_train_windowed = self.splitting_window(x_train, y_train, self.n_steps)
            train_data = tf.data.Dataset.from_tensor_slices(
                (x_train_windowed, y_train_windowed,)).shuffle(len(y_train_windowed)).batch(
                batch_size=BATCH_SIZE,
                drop_remainder=True, )

            x_test_windowed, y_test_windowed = self.splitting_window(x_test, y_test, self.n_steps)
            test_data = tf.data.Dataset.from_tensor_slices(
                (x_test_windowed, y_test_windowed,)).shuffle(len(y_test_windowed)).batch(
                batch_size=BATCH_SIZE,
                drop_remainder=True, )

            # predicting L2 average scores (regression)
            if self.target_is_language != 1:
                metrics = ["mean_squared_error"]
                n_outputs = 1
                loss_fn = tfk.losses.MeanSquaredError()
            # predicting L1 classes
            else:
                metrics = ["accuracy"]
                if len(np.unique(v['y_train'])) == 2:
                    loss_fn = tfk.losses.BinaryCrossentropy()
                    n_outputs = 1
                else:
                    loss_fn = tfk.losses.SparseCategoricalCrossentropy()
                    n_outputs = len(np.unique(v['y_train']))

            print(
                f" Predicting L1 or L2: {self.target_is_language}"
                f" Number of classes: {n_outputs} \n"
                f" x_train_windowed: {x_train_windowed.shape}",
                f"y_train_windowed: {y_train_windowed.shape} \n",
                f"x_test_windowed: {x_test_windowed.shape}",
                f"y_test_windowed: {y_test_windowed.shape} \n",
                f"train_data: {train_data.element_spec} \n",
                f"test_data: {test_data.element_spec} \n",
            )
            model = CNN1D(
                n_filters=64, n_features=v['x_train'].shape[1],
                kernel_size=3, n_outputs=n_outputs,
                n_steps=self.n_steps,
                target_is_language=self.target_is_language,
            )
            optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics,
            )
            start = time.time()
            history = model.fit(
                train_data,
                batch_size=BATCH_SIZE,
                epochs=self.n_epochs,
                verbose=2,
            )

            print(
                "summary of the model: \n ",
                model.summary()
            )

            x_test_, y_test_ = [], []
            for e in test_data.as_numpy_iterator():
                x_test_.append(e[0])
                y_test_.append(e[1])

            x_test_ = np.vstack(x_test_)
            y_test_ = np.hstack(y_test_)
            if self.target_is_language == 1:
                y_pred_prob = model.predict(test_data)
                y_pred = np.argmax(y_pred_prob, axis=-1)
            else:
                y_pred = model.predict(test_data)
                y_pred_prob = None

            print(
                f"x_test: {x_test_.shape} \n"
                f"y_test: {y_test_.shape} \n"
            )

            if y_pred_prob is not None:
                print(
                    f"y_pred_prob: {y_pred_prob.shape} \n"
                )

            end = time.time()
            self.results[k]["y_test"] = y_test_
            self.results[k]["x_test"] = x_test_
            self.results[k]["y_pred"] = np.squeeze(y_pred)
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

