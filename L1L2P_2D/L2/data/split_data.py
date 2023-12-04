import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

'''

requirements:

- split data by values of all specified columns
- return uneven or resampled data (stretching, truncation or other methods)
- resample into combined fixations on each word
- split data into train/test: randomly or test specific elements (Text_ID, SubjectID or others /
                                                                  n or a fraction of elements from specified columns)
- return data with or without fixation coordinates or demographics
- graph data: autocorrelation, correlation between languages...

'''

lang_list = ['du', 'ee', 'fi', 'ge', 'gr', 'he', 'it', 'no', 'ru', 'sp']
fix_cols = ['Fix_X', 'Fix_Y', 'Fix_Duration']
demo_cols = ['SubjectID', 'Text_ID', 'Language', 'motiv', 'IQ', 'Age', 'Sex']
data_types = {
    "Text_ID": int,
    "Fix_X": int,
    "Fix_Y": int,
    "Fix_Duration": int,
    "Word_Number": int,
    "Sentence": str,
    "Language": str,
    "SubjectID": str,
    "L2_spelling_skill": float,
    "L2_vocabulary_size": float,
    "vocab.t2.5": float,
    "L2_lexical_skill": float,
    "TOWRE_word": float,
    "TOWRE_nonword": float,
    "motiv": float,
    "IQ": float,
    "Age": int,
    "Sex": int,
    "Target_Ave": float,
    "Target_Label": int
}


def concat_MECO_langs(lang_names, path_to_data='../Data/csv/'):
    if lang_names == ['all']:
        lang_names = lang_list
    return pd.concat([pd.read_csv(f'{path_to_data}{lang}_fix_demo.csv') for lang in lang_names])


def split_into_rows(data, cols='fix+demo', target='Target_Label', with_values_only=None, shuffle_data=False):
    if cols == 'fix':
        cols = fix_cols
    elif cols == 'demo':
        cols = demo_cols
    elif cols == 'fix+demo':
        cols = fix_cols + demo_cols

    data = data.astype(data_types)
    if with_values_only is not None:
        for col, values in with_values_only.items():
            values = [data_types[col](value) for value in values]
            data = data[data[col].isin(values)]
    if cols == 'demo':
        data.drop_duplicates(subset=cols+[target], keep='first', inplace=True)

    X, y = data[cols], data[target]
    if shuffle_data:
        X, y = shuffle(X, y)
    return X, y


def truncate_series(X, y, demo, length):
    remove_indices = []
    for i in range(len(X)):
        if len(X[i]) >= length:
            X[i] = X[i][:length]
        else:
            remove_indices.append(i)
    for i in sorted(remove_indices, reverse=True):
        if i < len(X):
            X.pop(i)
            y.pop(i)
            demo.pop(i)
    return np.array(X), np.array(y), np.array(demo)


def get_series(data, target_row, labels, include_cols):
    X, y, demo = [], [], []
    combinations = np.array(np.meshgrid(*labels.values())).T.reshape(-1, len(labels))

    for values in combinations:
        condition_query = ''
        for col, value in zip(labels.keys(), values):
            if pd.api.types.is_numeric_dtype(data[col]):
                condition_query += f'({col} == {value}) & '
            else:
                value = value.replace("'", '`')
                condition_query += f'({col} == "{value}") & '
        cur_rows = data.query(condition_query[:-2])
        if not cur_rows.empty:
            X.append(cur_rows[include_cols])
            y.append(cur_rows[[target_row]].iloc[0])
            demo.append(cur_rows[demo_cols].iloc[0])
    return X, y, demo


def split_into_time_series(data, split_by=None, target='Target_Label',
                           th=0, truncate=False, length=180, test_size=0,
                           include_cols=None, train_labels=None, test_labels=None):
    if include_cols is None:
        include_cols = fix_cols
    if split_by is None:
        split_by = ['SubjectID', 'Text_ID']
    data = data.astype(data_types)
    data = data[data['Fix_Duration'] >= th]

    if train_labels is None:
        train_labels = dict(zip(split_by, [pd.unique(data[col]) for col in split_by]))
    X, y, demo = get_series(data, target, train_labels, include_cols)
    if truncate:
        X, y, demo = truncate_series(X, y, demo, length)

    if test_size == 0:
        return X, y, demo
    elif test_labels is not None:
        X_test, y_test, demo_test = get_series(data, target, test_labels, include_cols)
        if truncate:
            X_test, y_test, demo_test = truncate_series(X_test, y_test, demo_test, length)
    else:
        X, X_test, y, y_test, demo, demo_test = train_test_split(X, y, demo, test_size=test_size)
    return X, X_test, y, y_test, demo, demo_test


def make_dataset(lang_names=None, data_type='fix', target='Target_Label', params=None,
                 cv_col='SubjectID', path_to_data='Data/csv/'):
    if lang_names is None:
        lang_names = lang_list
    data = concat_MECO_langs(lang_names, path_to_data=path_to_data)
    X, y = split_into_rows(data, cols=data_type, target=target, with_values_only=params)

    if cv_col is not None:
        keys = data[cv_col].unique()
        cv_dict = {key: num for num, key in enumerate(keys)}
        cv_col = np.vectorize(cv_dict.get)(data[cv_col].to_numpy())
    return X, y, cv_col
