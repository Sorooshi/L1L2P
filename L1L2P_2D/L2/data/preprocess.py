import numpy as np
import pandas as pd


def convert_range(x_old, old_min, old_max, new_min, new_max):
    x_new = ((new_max - new_min) * (x_old - old_min)) / (old_max - old_min) + new_min
    return x_new


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
            tmp2 = pd.concat([tmp2] * 1, ignore_index=True)
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


class L2DataPreprocessor:
    def __init__(self,
                 language_name,
                 path_to_load_data,
                 path_to_save_data='',
                 print_debug=False):

        self.language_name = language_name
        self.path_to_save_data = path_to_save_data
        self.print_debug = print_debug

        self.age_gender = pd.read_excel(path_to_load_data + 'age_gender.xlsx').dropna()
        self.check_age_gender_data()
        self.age_gender_subject_id = set(self.age_gender.SubjectID)

        self.demo_all_lang = pd.read_excel(path_to_load_data + 'demo_alllang.xlsx').dropna()
        self.check_demo_all_lang()
        self.demo_all_lang_subject_id = set(self.demo_all_lang.SubjectID)

        self.L2_scores = {
            "L2_spelling_skill": [0, 44],
            "L2_vocabulary_size": [0, 100],
            "vocab.t2.5": [0, 40],  # still not clear but assume it is 0, 100
            "L2_lexical_skill": [0, 100],
            "TOWRE_word": [0, 104],
            "TOWRE_nonword": [0, 63],
        }
        self.rescale_l2_scores()

        self.lang_data = pd.read_excel(path_to_load_data + f'CLEAN_DATA/{language_name}_data.xlsx').dropna()
        self.check_lang_data()

        subid_dict = dict(zip(self.demo_all_lang.subid, self.demo_all_lang.SubjectID))
        self.lang_data['SubjectID'].map(subid_dict)
        self.lang_data_subject_id = set(self.lang_data.SubjectID)

        self.age_gender_lang = self.age_gender.loc[self.age_gender.lang == language_name]
        self.age_gender_lang_subject_id = set(self.age_gender_lang.SubjectID)

        self.demo_lang = self.demo_all_lang.loc[self.demo_all_lang.lang == language_name]
        self.demo_lang_subject_id = set(self.demo_lang.SubjectID)

    def check_age_gender_data(self):
        for index, row in self.age_gender.iterrows():
            if not (
                    isinstance(row['SubjectID'], str) and
                    isinstance(row['Age'], int, ) and
                    isinstance(row['lang'], str)
            ):
                print(
                    "Inconsistencies! \n"
                    f"index:{index} \t SubjectID:{row['SubjectID'], type(row['SubjectID'])}  "
                    f"Age:{row['Age'], type(row['Age'])}   Lang:{row['lang'], type(row['lang'])}"
                )
        self.age_gender = self.age_gender.astype(({
            "SubjectID": str,
            "Age": int,
            "Sex": int,
            "lang": str,
        }))

    def check_demo_all_lang(self):
        if self.print_debug:
            for index, row in self.demo_all_lang.iterrows():
                if not (isinstance(row['SubjectID'], str) and
                        isinstance(row['lang'], str) and
                        isinstance(row['L2_spelling_skill'], float) and
                        isinstance(row['L2_vocabulary_size'], float) and
                        isinstance(row['vocab.t2.5'], float) and
                        isinstance(row['L2_lexical_skill'], float) and

                        isinstance(row['TOWRE_word'], float) and

                        isinstance(row['TOWRE_nonword'], float) and

                        isinstance(row['motiv'], float) and
                        isinstance(row['IQ'], float)):
                    print(f"Inconsistencies in index:{index} \t SubjectID:{row['SubjectID']}", row)

        self.demo_all_lang = self.demo_all_lang.astype({
            "subid": str,
            "SubjectID": str,
            "lang": str,
            "L2_spelling_skill": float,
            "L2_vocabulary_size": float,
            "vocab.t2.5": float,
            "L2_lexical_skill": float,
            "TOWRE_word": float,
            "TOWRE_nonword": float,
            "motiv": float,
            "IQ": float,
        })

    def check_lang_data(self):
        if self.print_debug:
            for index, row in self.lang_data.iterrows():
                if not (
                        isinstance(row['SubjectID'], str) and
                        isinstance(row['Text_ID'], int) and
                        isinstance(row['Fix_X'], int) and
                        isinstance(row['Fix_Y'], int) and
                        isinstance(row['Fix_Duration'], int) and
                        isinstance(row['Word'], str) and
                        isinstance(row['Sentence'], str) and
                        isinstance(row['Language'], str)
                ):
                    print(
                        "Inconsistencies! \n"
                        f"index:{index} \t SubjectID:{row['SubjectID'],} \n",
                        isinstance(row['SubjectID'], str), "\n",
                        isinstance(row['Text_ID'], int), "\n",
                        isinstance(row['Fix_X'], int), "\n",
                        isinstance(row['Fix_Y'], int), "\n",
                        isinstance(row['Fix_Duration'], int), "\n",
                        isinstance(row['Word'], str), "\n",
                        isinstance(row['Sentence'], str), "\n",
                        isinstance(row['Language'], str), "\n",
                    )
        self.lang_data = self.lang_data.astype({
            "SubjectID": str,
            "Text_ID": int,
            "Fix_X": int,
            "Fix_Y": int,
            "Fix_Duration": int,
            "Word_Number": int,
            "Sentence": str,
            "Language": str,
        })

    def rescale_l2_scores(self):
        if self.print_debug:
            for k, v in self.L2_scores.items():
                print(k, f"Lower={v[0]}, Upper={v[1]}")

        for k, v in self.L2_scores.items():
            self.demo_all_lang[k] = self.demo_all_lang[k].apply(convert_range, args=(v[0], v[1], 0, 5))

        cols = list(self.L2_scores.keys())

        # demo["Target_Ave"] = demo[cols].sum(axis=1)/len(cols)
        self.demo_all_lang["Target_Ave"] = self.demo_all_lang[cols].apply(np.average, axis=1)
        self.demo_all_lang["Target_Label"] = self.demo_all_lang["Target_Ave"].apply(np.round).values

    def print_missing_data(self):
        print(
            f" age-gender =  {len(self.age_gender_lang_subject_id)} \n",
            f"ru_demo    =  {len(self.demo_lang_subject_id)} \n",
            f"fixation   = {len(self.lang_data_subject_id)} \n\n",

            f" missing in age_gender or demo    : "
            f"{self.age_gender_lang_subject_id.symmetric_difference(self.demo_lang_subject_id)} \n",
            f"missing in age_gender or fixation: "
            f"{self.age_gender_lang_subject_id.symmetric_difference(self.lang_data_subject_id)} \n",
            f"missing in fixation or demo      : "
            f"{self.lang_data_subject_id.symmetric_difference(self.demo_lang_subject_id)} \n\n",

            f" missing in demo vs age_gender    : "
            f"{self.demo_lang_subject_id.difference(self.age_gender_lang_subject_id)} \n",
            f"missing in fixation vs age_gender:"
            f" {self.lang_data_subject_id.difference(self.age_gender_lang_subject_id)} \n\n",

            f" missing in age_gender vs demo   : "
            f"{self.age_gender_lang_subject_id.difference(self.demo_lang_subject_id)} \n",
            f"missing in age_gender vs fixation: "
            f"{self.age_gender_lang_subject_id.difference(self.lang_data_subject_id)} \n\n",

            f" missing in fixation vs demo   : "
            f"{self.lang_data_subject_id.difference(self.demo_lang_subject_id)} \n",
            f"missing in demo vs fixation   : "
            f"{self.demo_lang_subject_id.difference(self.lang_data_subject_id)} \n\n",
        )

    def midterm_conclusion(self):
        subject_to_keep = self.age_gender_lang_subject_id.intersection(
            self.demo_lang_subject_id).intersection(self.lang_data_subject_id)
        lang_age_gender = self.age_gender_lang.loc[self.age_gender_lang.SubjectID.isin(subject_to_keep)]
        lang_demo = self.demo_lang.loc[self.demo_lang.SubjectID.isin(subject_to_keep)]
        lang_fix_data = self.lang_data.loc[self.lang_data.SubjectID.isin(subject_to_keep)]

        demo_lang_concat = concat_dfs(
            df1=lang_demo, df2=lang_age_gender,
            features1=[
                'SubjectID', 'lang', 'L2_spelling_skill', 'L2_vocabulary_size',
                'vocab.t2.5', 'L2_lexical_skill', 'TOWRE_word', 'TOWRE_nonword',
                'motiv', 'IQ', 'Target_Ave', 'Target_Label'],
            features2=["Age", "Sex"],
        )

        lang_fix_demo = concat_dfs(
            df1=lang_fix_data, df2=demo_lang_concat,
            features1=[
                'Text_ID', 'Fix_X', 'Fix_Y', 'Fix_Duration',
                'Word_Number', 'Word', 'Sentence', 'Language'],
            features2=[
                'SubjectID', 'L2_spelling_skill', 'L2_vocabulary_size',
                'vocab.t2.5', 'L2_lexical_skill', 'TOWRE_word', 'TOWRE_nonword',
                'motiv', 'IQ', 'Age', 'Sex', 'Target_Ave', 'Target_Label', ],
        )

        lang_fix_demo.to_csv(self.path_to_save_data + f'{self.language_name}_fix_demo.csv', index=False)
