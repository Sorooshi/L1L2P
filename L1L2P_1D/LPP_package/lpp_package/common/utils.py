import os
#import wandb
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


np.set_printoptions(suppress=True, precision=3)


def save_a_dict(a_dict, name, save_path, ):
    with open(os.path.join(save_path, name+".pickle"), "wb") as fp:
        pickle.dump(a_dict, fp)
    return None


def load_a_dict(name, save_path, ):
    with open(os.path.join(save_path, name + ".pickle"), "rb") as fp:
        a_dict = pickle.load(fp)
    return a_dict


def mae(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true-y_pred))


def rmse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean(np.power(y_true-y_pred, 2)))


def mrae(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
    return np.mean(np.abs(np.divide(y_true - y_pred, y_true)))


def jsd(y_true, y_pred):
    return np.asarray(distance.jensenshannon(y_true, y_pred))


def mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100):
    errors = []
    inds = np.arange(len(y_true))

    for i in range(n_iters):
        inds_boot = resample(inds)

        y_true_boot = y_true[inds_boot]
        y_pred_boot = y_pred[inds_boot]

        y_true_mean = y_true_boot.mean(axis=0)
        y_pred_mean = y_pred_boot.mean(axis=0)

        ierr = np.abs((y_true_mean - y_pred_mean) / y_true_mean) * 100
        errors.append(ierr)

    errors = np.array(errors)
    return errors


def discrepancy_score(observations, forecasts, model='QDA', n_iters=1):

    """
    Parameters:
    -----------
    observations : numpy.ndarray, shape=(n_samples, n_features)
        True values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    forecasts : numpy.ndarray, shape=(n_samples, n_features)
        Predicted values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    model : sklearn binary classifier
        Possible values: RF, DT, LR, QDA, GBDT
    n_iters : int
        Number of iteration per one forecast.

    Returns:
    --------
    mean : float
        Mean value of discrepancy score.
    std : float
        Standard deviation of the mean discrepancy score.

    """

    scores = []

    X0 = observations
    y0 = np.zeros(len(observations))

    X1 = forecasts
    y1 = np.ones(len(forecasts))

    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)

    for it in range(n_iters):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)
        if model == 'RF':
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=None)
        elif model == 'GDBT':
            clf = GradientBoostingClassifier(max_depth=6, subsample=0.7)
        elif model == 'DT':
            clf = DecisionTreeClassifier(max_depth=10)
        elif model == 'LR':
            clf = LogisticRegression()
        elif model == 'QDA':
            clf = QuadraticDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        y_pred_test = clf.predict_proba(x_test)[:, 1]
        auc = 2 * metrics.roc_auc_score(y_test, y_pred_test) - 1
        scores.append(auc)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std() / np.sqrt(len(scores))

    return mean, std


def evaluate_a_x_test(y_true, y_pred,):

    # MEAPE >> Mean Estimation Absolute Percentage Error
    meape_mu, meape_std = mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100)

    # gb >> Gradient Decent Boosting Classifier
    gb_mu, gb_std = discrepancy_score(y_true, y_pred, model='GDBT', n_iters=10)

    # qda >> Quadratic Discriminant Analysis
    qda_mu, qda_std = discrepancy_score(y_true, y_pred, model='QDA', n_iters=10)

    return meape_mu, gb_mu, qda_mu


def save_model(path, model, specifier, ):
    dump(
        model, os.path.join(
            path, specifier+".joblib"
        )
    )
    return None


def print_the_evaluated_results(results, learning_method, ):
    """ results: dict, containing results of each repeat, key:= repeat number.
            learning_method: string, specifing which metrics should be used.
    """

    # Regression metrics
    MEA, RMSE, MRAE, JSD, R2_Score, MEAPE_mu, MEAPE_std = [], [], [], [], [], [], []
    # Classification and clustering metrics
    ARI, NMI, Precision, Recall, F1_Score, ROC_AUC, ACC, TNR = [], [], [], [], [], [], [], []

    for repeat, result in results.items():
        y_true = result["y_test"]
        y_pred = result["y_pred"]
        try:
            y_pred_prob = result["y_pred_prob"]
        except:
            y_pred_prob = None
            print("No prediction probability exist.")

        if learning_method == "regression":

            MEA.append(mae(y_true=y_true, y_pred=y_pred))
            RMSE.append(rmse(y_true=y_true, y_pred=y_pred))
            MRAE.append(mrae(y_true=y_true, y_pred=y_pred))
            JSD.append(jsd(y_true=y_true, y_pred=y_pred).mean())
            R2_Score.append(metrics.r2_score(y_true, y_pred))
            meape_errors = mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100
            )
            MEAPE_mu.append(meape_errors.mean(axis=0))
            MEAPE_std.append(meape_errors.std(axis=0))

        else:
            ARI.append(metrics.adjusted_rand_score(y_true, y_pred))
            NMI.append(metrics.normalized_mutual_info_score(y_true, y_pred))
            JSD.append(jsd(y_true=y_true, y_pred=y_pred).mean())
            Precision.append(metrics.precision_score(y_true, y_pred, average='weighted'))
            Recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
            F1_Score.append(metrics.f1_score(y_true, y_pred, average='weighted'))
            meape_errors = mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100
            )
            MEAPE_mu.append(meape_errors.mean(axis=0))
            MEAPE_std.append(meape_errors.std(axis=0))
            ACC.append(metrics.accuracy_score(y_true, y_pred, ))

        if learning_method == "classification":

            # to compute ROC_AUC
            try:
                y_true.shape[1]
                y_true_ = y_true
                print(
                    f"y_true roc auc: {y_true_.shape} \n"
                    f"y_pred roc auc: {y_pred.shape} \n"
                    f"y_pred_prob roc auc: {y_pred_prob.shape} \n"
                )
            except:
                enc = OneHotEncoder(sparse=False)
                y_true_ = y_true.reshape(-1, 1)
                y_true_ = enc.fit_transform(y_true_)
                # baseline uncomment it:
                # y_pred_prob = np.repeat(y_pred_prob, y_true_.shape[1]).reshape(-1, y_true_.shape[1])
                print(
                    f"y_true_: {y_true_} \n"
                    f"y_true: {y_true} \n"
                    f"y_pred: {y_pred} \n"
                    f"y_pred_prob: {y_pred_prob} \n"
                    f"y_true roc auc conv.: {y_true_.shape} \n"
                    f"y_pred roc auc conv.: {y_pred.shape} \n"
                    f"y_pred_prob roc auc: {y_pred_prob.shape} \n"
                )

            if y_pred_prob is not None:
                ROC_AUC.append(
                    metrics.roc_auc_score(y_true_, y_pred_prob, average='weighted', multi_class="ovr"),
                )

            cm = metrics.confusion_matrix(y_true, y_pred, )
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            tnr = tn.astype(float) / (tn.astype(float) + fp.astype(float))
            _, support = np.unique(y_true, return_counts=True)
            tnr = np.dot(tnr, support) / sum(support)
            TNR.append(tnr)

        elif learning_method == "abnormality_detection":
            # to compute ROC_AUC
            # try:
            #     y_true.shape[1]
            #     y_true_ = y_true
            # except:
            #     enc = OneHotEncoder(sparse=False)
            #     y_true_ = y_true.reshape(-1, 1)
            #     y_true_ = enc.fit_transform(y_true_)

            if y_pred_prob is not None:
                ROC_AUC.append(
                    metrics.roc_auc_score(y_true, y_pred_prob,
                                          average='weighted',
                                          ),
                )

            cm = metrics.confusion_matrix(y_true, y_pred,)
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            tnr = tn.astype(float) / (tn.astype(float) + fp.astype(float))
            _, support = np.unique(y_true, return_counts=True)
            tnr = np.dot(tnr, support)/sum(support)
            TNR.append(tnr)

        else:
            ROC_AUC.append(123456)  # appending an impossible outcome of ROC_AUC to avoid adding one more

    if learning_method == "regression":
        MEA = np.nan_to_num(np.asarray(MEA))
        RMSE = np.nan_to_num(np.asarray(RMSE))
        MRAE = np.nan_to_num(np.asarray(MRAE))
        JSD = np.nan_to_num(np.asarray(JSD))
        R2_Score = np.nan_to_num(np.asarray(R2_Score))
        MEAPE_mu = np.nan_to_num(np.asarray(MEAPE_mu))

        mae_ave = np.mean(MEA, axis=0)
        mae_std = np.std(MEA, axis=0)

        rmse_ave = np.mean(RMSE, axis=0)
        rmse_std = np.std(RMSE, axis=0)

        mrae_ave = np.mean(MRAE, axis=0)
        mrae_std = np.std(MRAE, axis=0)

        jsd_ave = np.mean(JSD, axis=0)
        jsd_std = np.std(JSD, axis=0)

        r2_ave = np.mean(R2_Score, axis=0)
        r2_std = np.std(R2_Score, axis=0)

        meape_ave = np.mean(MEAPE_mu, axis=0)
        meape_std = np.std(MEAPE_mu, axis=0)

        print("   mae ", "   rmse ", "\t mrae",
              "\t r2_score ", "\t meape ", "\t jsd ",
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ",
              )

        print(
            "%.3f" % mae_ave, "%.3f" % mae_std,
            "%.3f" % rmse_ave, "%.3f" % rmse_std,
            "%.3f" % mrae_ave, "%.3f" % mrae_std,
            "%.3f" % r2_ave, "%.3f" % r2_std,
            "%.3f" % meape_ave, "%.3f" % meape_std,
            "%.3f" % jsd_ave, "%.3f" % jsd_std,
        )

    else:
        JSD = np.nan_to_num(np.asarray(JSD))
        MEAPE_mu = np.nan_to_num(np.asarray(MEAPE_mu))
        ARI = np.nan_to_num(np.asarray(ARI))
        NMI = np.nan_to_num(np.asarray(NMI))
        Precision = np.nan_to_num(np.asarray(Precision))
        Recall = np.nan_to_num(np.asarray(Recall))
        F1_Score = np.nan_to_num(np.asarray(F1_Score))
        ROC_AUC = np.nan_to_num(np.asarray(ROC_AUC))
        ACC = np.nan_to_num(np.asarray(ACC))
        TNR = np.nan_to_num(np.asarray(TNR))

        ari_ave = np.mean(ARI, axis=0)
        ari_std = np.std(ARI, axis=0)

        nmi_ave = np.mean(NMI, axis=0)
        nmi_std = np.std(NMI, axis=0)

        precision_ave = np.mean(Precision, axis=0)
        precision_std = np.std(Precision, axis=0)

        recall_ave = np.mean(Recall, axis=0)
        recall_std = np.std(Recall, axis=0)

        f1_score_ave = np.mean(F1_Score, axis=0)
        f1_score_std = np.std(F1_Score, axis=0)

        roc_auc_ave = np.mean(ROC_AUC, axis=0)
        roc_auv_std = np.std(ROC_AUC, axis=0)

        jsd_ave = np.mean(JSD, axis=0)
        jsd_std = np.std(JSD, axis=0)

        meape_ave = np.mean(MEAPE_mu, axis=0)
        meape_std = np.std(MEAPE_std, axis=0)

        acc_ave = np.mean(ACC, axis=0)
        acc_std = np.std(ACC, axis=0)

        tnr_ave = np.mean(TNR, axis=0)
        tnr_std = np.std(TNR, axis=0)

        print("  ari ", "  nmi ", "\t preci", "\t recall ",
                  "\t f1_score ", "\t roc_auc ", "\t meape ", "\t jsd ", "\t acc", "\t tnr"
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std "
              )

        print("%.3f" % ari_ave, "%.3f" % ari_std,
              "%.3f" % nmi_ave, "%.3f" % nmi_std,
              "%.3f" % precision_ave, "%.3f" % precision_std,
              "%.3f" % recall_ave, "%.3f" % recall_std,
              "%.3f" % f1_score_ave, "%.3f" % f1_score_std,
              "%.3f" % roc_auc_ave, "%.3f" % roc_auv_std,
              "%.3f" % meape_ave, "%.3f" % meape_std,
              "%.3f" % jsd_ave, "%.3f" % jsd_std,
              "%.3f" % acc_ave, "%.3f" % acc_std,
              "%.3f" % tnr_ave, "%.3f" % tnr_std,
              )

    return None





