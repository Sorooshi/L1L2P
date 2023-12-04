from keras import layers as ll
from keras.models import Model

from common.utils import get_optimizers_for_system
from common.metrics import root_mean_squared_error

NUM_CLASSES = 10
IMG_SHAPE = (60, 180, 1)
DEMO_SHAPE = (5,)
allowed_models = ['base_cnn', 'multiinput_cnn']
allowed_methods = ['clf', 'reg']


def build_base_cnn(activation: str, pool_dim: int, conv1_filter: int, conv1_kernel: int,
                   conv2_filter: int, conv2_kernel: int, conv3_filter: int, conv3_kernel: int,
                   dropout: int, dense: int, optimizer: str, lr: float, method='clf', **kwargs):
    
    pool_size = (pool_dim, pool_dim)
    if method == 'clf':
        output_layer = ll.Dense(NUM_CLASSES, activation='softmax')
        loss = 'sparse_categorical_crossentropy'
    else:
        output_layer = ll.Dense(1, activation='linear')
        loss = root_mean_squared_error
    
    img = ll.Input(shape=IMG_SHAPE, name='img')
    x = ll.Conv2D(conv1_filter, conv1_kernel, padding='same', activation=activation)(img)
    x = ll.Dropout(dropout)(x)
    x = ll.BatchNormalization()(x)
    x = ll.Conv2D(conv2_filter, conv2_kernel, padding='same', activation=activation)(x)
    x = ll.Dropout(dropout)(x)
    x = ll.MaxPooling2D(pool_size, padding='same')(x)
    x = ll.Conv2D(conv3_filter, conv3_kernel, padding='same', activation=activation)(x)
    x = ll.Dropout(dropout)(x)
    x = ll.MaxPooling2D(pool_size, padding='same')(x)
    x = ll.BatchNormalization()(x)
    x = ll.Flatten()(x)
    x = ll.Dense(dense, activation=activation)(x)
    out = output_layer(x)

    model = Model(inputs=[img], outputs=out)
    optimizer = get_optimizers_for_system()[optimizer]
    model.compile(optimizer(lr), loss, run_eagerly=True)
    return model


def build_base_cnn_clf(**kwargs):
    return build_base_cnn(method='clf', **kwargs)


def build_base_cnn_reg(**kwargs):
    return build_base_cnn(method='reg', **kwargs)


def build_multiinput_cnn(activation: str, pool_dim: int, conv1_filter: int, conv1_kernel: int,
                         conv2_filter: int, conv2_kernel: int, conv3_filter: int, conv3_kernel: int,
                         dropout: int, dense_img: int, dense_mlp: int, dense_concat: int,
                         optimizer: str, lr: float, method='clf', **kwargs):
    pool_size = (pool_dim, pool_dim)
    if method == 'clf':
        output_layer = ll.Dense(NUM_CLASSES, activation='softmax')
        loss = 'sparse_categorical_crossentropy'
    else:
        output_layer = ll.Dense(1, activation='linear')
        loss = root_mean_squared_error

    # define two sets of inputs
    img = ll.Input(shape=IMG_SHAPE, name='img')
    demo = ll.Input(shape=DEMO_SHAPE, name='demo')

    # build CNN part
    x = ll.Conv2D(conv1_filter, conv1_kernel, padding='same', activation=activation)(img)
    x = ll.Dropout(dropout)(x)
    x = ll.BatchNormalization()(x)
    x = ll.Conv2D(conv2_filter, conv2_kernel, padding='same', activation=activation)(x)
    x = ll.Dropout(dropout)(x)
    x = ll.MaxPooling2D(pool_size, padding='same')(x)
    x = ll.Conv2D(conv3_filter, conv3_kernel, padding='same', activation=activation)(x)
    x = ll.Dropout(dropout)(x)
    x = ll.MaxPooling2D(pool_size, padding='same')(x)
    x = ll.BatchNormalization()(x)
    x = ll.Flatten()(x)
    x = ll.Dense(dense_img, activation=activation)(x)
    x = ll.Dense(dense_img, activation=activation)(x)

    # build MLP part
    y = ll.Dense(dense_mlp, activation=activation)(demo)
    y = ll.Dense(dense_mlp, activation=activation)(y)
    
    # combine the two parts
    out = ll.concatenate([x, y])
    out = ll.Dense(dense_concat, activation=activation)(out)
    out = output_layer(out)

    # compile model
    model = Model(inputs=[img, demo], outputs=out)
    optimizer = get_optimizers_for_system()[optimizer]
    model.compile(optimizer(lr), loss, run_eagerly=True)
    return model


def build_multiinput_cnn_clf(**kwargs):
    return build_multiinput_cnn(method='clf', **kwargs)


def build_multiinput_cnn_reg(**kwargs):
    return build_multiinput_cnn(method='reg', **kwargs)


build_model = {
    'base_cnn': {
        'clf': build_base_cnn_clf,
        'reg': build_base_cnn_reg
    },
    'multiinput_cnn': {
        'clf': build_multiinput_cnn_clf,
        'reg': build_multiinput_cnn_reg
    }
}
