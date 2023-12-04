import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split


class LanguageData:
    """" various forms of dataset(s)  """

    def __init__(self,
                 n_splits: int = 5,
                 n_repeats: int = 10,
                 path: Path = Path("../datasets_LPP"),
                 data_name: str = "ru_fix_demo",
                 ):

        self.path = path
        self.ru_fix_data_name = data_name
        self.all_fix_data_name = data_name
        self.ru_demo_data_name = data_name
        self.all_demo_data_name = data_name
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self.csv_ru_fix_demo = pd.read_csv(os.path.join(self.path, self.ru_fix_data_name + ".csv"))
        self.csv_all_fix_demo = pd.read_csv(os.path.join(self.path, self.all_fix_data_name + ".csv"))
        self.csv_ru_demo = pd.read_csv(os.path.join(self.path, self.ru_demo_data_name + ".csv"))
        self.csv_all_demo = pd.read_csv(os.path.join(self.path, self.all_demo_data_name + ".csv"))
        self.fix_demo_datasets = defaultdict(list)

        self.x = pd.DataFrame()  # features/random variables (either shuffled or not)
        self.y = pd.DataFrame()  # targets variables/predictions (in corresponding to x)
        self.x_dum_test_pp_df = pd.DataFrame()  # independent/real-world test data

        self.stratified_kFold_cv = None
        self.stratified_train_test_splits = defaultdict(defaultdict)
        self.features = None

    def get_ru_fix_demo_datasets(self, ):
        print("Loading Fix_Demo data from csv: ")
        tmp = self.csv_ru_fix_demo
        tmp = tmp.astype({
            "Text_ID": str,
            "Fix_X": int,
            "Fix_Y": int,
            "Fix_Duration": int,
            "Word_Number": int,
            "SubjectID": str,
            "L2_spelling_skill": float,
            "L2_vocabulary_size": float,
            "vocab.t2.5": float,
            "L2_lexical_skill": float,
            "TOWRE_word": float,
            "TOWRE_nonword": float,
            "motiv": float,
            "IQ": int,
            "Age": int,
            "Sex": int,
            "Target_Label": int,
            "Target_Ave": float
        })
        self.fix_demo_datasets = tmp
        print(" ",self.fix_demo_datasets.shape)
        print(" ")
        return self.fix_demo_datasets

    def get_all_fix_demo_datasets(self, ):
        tmp = self.csv_all_fix_demo
        tmp = tmp.astype({
            "Text_ID": str,
            "Fix_X": int,
            "Fix_Y": int,
            "Fix_Duration": int,
            "Word_Number": int,
            "SubjectID": str,
            "L2_spelling_skill": float,
            "L2_vocabulary_size": float,
            "vocab.t2.5": float,
            "L2_lexical_skill": float,
            "TOWRE_word": float,
            "TOWRE_nonword": float,
            "motiv": float,
            "IQ": int,
            "Age": int,
            "Sex": int,
            "Target_Label": int,
            "Target_Ave": float
        })
        self.fix_demo_datasets = tmp
        return self.fix_demo_datasets

    def get_ru_demo_datasets(self, ):
        tmp = self.csv_ru_demo
        tmp = tmp.astype({
            "Text_ID": str,
            "Fix_X": int,
            "Fix_Y": int,
            "Fix_Duration": int,
            "Word_Number": int,
            "SubjectID": str,
            "L2_spelling_skill": float,
            "L2_vocabulary_size": float,
            "vocab.t2.5": float,
            "L2_lexical_skill": float,
            "TOWRE_word": float,
            "TOWRE_nonword": float,
            "motiv": float,
            "IQ": int,
            "Age": int,
            "Sex": int,
            "Target_Label": int,
            "Target_Ave": float
        })
        self.demo_datasets = tmp
        return self.demo_datasets

    def get_all_demo_datasets(self, ):
        tmp = self.csv_all_demo
        tmp = tmp.astype({
            "Text_ID": str,
            "Fix_X": int,
            "Fix_Y": int,
            "Fix_Duration": int,
            "Word_Number": int,
            "SubjectID": str,
            "L2_spelling_skill": float,
            "L2_vocabulary_size": float,
            "vocab.t2.5": float,
            "L2_lexical_skill": float,
            "TOWRE_word": float,
            "TOWRE_nonword": float,
            "motiv": float,
            "IQ": int,
            "Age": int,
            "Sex": int,
            "Target_Label": int,
            "Target_Ave": float
        })
        self.demo_datasets = tmp
        return self.demo_datasets

    def get_all_fix_demo_language_datasets(self, ):
        tmp = self.csv_all_fix_demo
        tmp = tmp.astype({
            "Text_ID": str,
            "Fix_X": int,
            "Fix_Y": int,
            "Fix_Duration": int,
            "Word_Number": int,
            "SubjectID": str,
            "L2_spelling_skill": float,
            "L2_vocabulary_size": float,
            "vocab.t2.5": float,
            "L2_lexical_skill": float,
            "TOWRE_word": float,
            "TOWRE_nonword": float,
            "motiv": float,
            "IQ": int,
            "Age": int,
            "Sex": int,
            "Target_Label": int,
            "Target_Ave": float,
            "Language": str,
        })
        tmp.replace(
            to_replace={"Language": {"du": 1, "ee": 2, "fi": 3, "ge": 4, "gr": 5, "he": 6,
                                     "it": 7, "no": 8, "ru": 9, "sp": 10, "tr": 11}},
            inplace=True,
        )

        self.demo_datasets = tmp
        return self.demo_datasets

    def get_onehot_features_targets(self, data, c_features=None, indicators=None, targets=None):
        """ Returns x, y, pd.DataFrames, of features and targets values respectively. """
        if c_features:
            data = pd.get_dummies(
                data=data, columns=c_features
            )

        if not indicators:
            indicators = []
        if not targets:
            targets = ['Target_Ave', 'Target_Label']

        self.features = list(
            set(data.columns).difference(
                set(indicators).union(set(targets))
            )
        )

        self.x = data.loc[:, self.features]
        self.y = data.loc[:, targets]

        return self.x, self.y

    def get_stratified_kfold_cv(self, to_shuffle, n_splits):

        """ Returns a CV object to be used in Bayesian/Grid/Random
        search optimization to tune the estimator(s) hyper-parameters.
        """
        self.stratified_kFold_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=to_shuffle
        )

        return self.stratified_kFold_cv

    def get_stratified_train_test_splits(self, x, y, labels, to_shuffle=True, n_splits=10):
        """ Returns dict containing repeated train and test splits.
                Repeat numbers are separated from the rest of strinds in the key with a single dash "-".
        """
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=to_shuffle
        )

        repeat = 0
        for train_index, test_index in skf.split(x, labels):  # labels := y.Group: to provide correct stratified splits
            repeat += 1
            k = str(repeat)
            self.stratified_train_test_splits[k] = defaultdict(list)
            self.stratified_train_test_splits[k]["x_train"] = x[train_index]
            self.stratified_train_test_splits[k]["x_test"] = x[test_index]
            self.stratified_train_test_splits[k]["y_train"] = y[train_index]
            self.stratified_train_test_splits[k]["y_test"] = y[test_index]

        return self.stratified_train_test_splits

    def _get_sub_categories_quant_stats(self, ):

        self.sub_categories = {f: [] for f in self.features}
        if self.c_features:
            for f in self.c_features:
                pattern = re.compile(f)
                str_match = [x for x in self.features_dum if re.search(f, x)]
                self.sub_categories[f] = str_match

        return self.sub_categories

    @staticmethod
    def _remove_missing_data(df):
        for col in df.columns:
            try:
                df[col].replace({".": np.nan}, inplace=True)
            except Exception as e:
                print(e, "\n No missing values in", col)

        return df.dropna()

    @staticmethod
    def concat_dfs(df1, df2, features1, features2):

        """ concatenates df2 to df1, that is, it casts df2's dimensions df1. """

        data = []
        subject_ids = df2.SubjectID
        for subject_id in subject_ids:
            tmp1 = df1.loc[(df1.SubjectID == subject_id)]
            tmp1 = tmp1.loc[:, features1].reset_index(drop=True)
            tmp2 = df2.loc[df2.SubjectID == subject_id]
            tmp2 = tmp2.loc[:, features2]

            n = tmp1.shape[0]
            if n == tmp2.shape[0]:
                tmp2 = pd.concat([tmp2], ignore_index=True)
            else:
                tmp2 = pd.concat([tmp2] * n, ignore_index=True)  # .reset_index(drop=True)

            tmp3 = pd.concat([tmp1, tmp2], axis=1, )

            if tmp3.shape[0] != tmp1.shape[0] or tmp3.shape[0] != tmp2.shape[0]:
                print(
                    subject_id,
                    "in consistencies in number of observations (rows)"
                )

            if tmp3.shape[1] != tmp1.shape[1] + tmp2.shape[1]:
                print(
                    subject_id,
                    "inconsistencies in feature space (columns)"
                )

            data.append(tmp3)

        return pd.concat(data)




