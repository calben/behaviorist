import pandas as pd
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import cross_validation, linear_model
import sklearn.metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def load_feature_matrix_for_ml(csv: str) -> dict:
    df = pd.read_csv(csv, index_col=0)
    df = filter_feature_matrix(df)

    data = {}
    data["target"] = df["label"].values
    data["features"] = df.drop("label", axis=1).values
    return data


def filter_feature_matrix(mat: pd.DataFrame, balance_input=True) -> pd.DataFrame:
    mat = mat.dropna(axis=0)
    mat = mat[mat["distance-mean"] != 0]
    if balance_input:
        mat_pos = mat[mat["label"] == 1]
        mat_neg = mat[mat["label"] == 0]
        print(len(mat_pos), "positive samples")
        print(len(mat_neg), "negative samples")
        if len(mat_pos) >= len(mat_neg):
            mat_pos = mat_pos.sample(len(mat_neg))
        else:
            mat_neg = mat_neg.sample(len(mat_pos))
    mat = pd.concat([mat_pos, mat_neg])
    mat = mat.reset_index(drop=True)
    mat = mat[["label", "distance-std", "distance-25%", "distance-75%", "correlation-abs"]]
    return mat


def run_cross_validation(data: dict):
    x_train, x_test, y_train, y_test = \
        cross_validation.train_test_split(data["features"], data["target"],
                                          test_size=0.4,
                                          random_state=0)
    clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    return clf.score(x_test, y_test)


def run_k_cross_validation(data: dict):
    X = data["features"]
    y = data["target"]
    clf = svm.SVC()
    kf = KFold(len(X), n_folds=len(X))
    svm_scores = []
    svm_predictions = []
    for k, (train, test) in enumerate(kf):
        clf.fit(X[train], y[train])
        svm_score = clf.score(X[test], y[test])
        svm_prediction = clf.predict(X[test])
        svm_scores.append(svm_score)
        svm_predictions.append(svm_prediction)

    svm_predictions = np.asarray(svm_predictions)

    print("F1:", str(sklearn.metrics.f1_score(y, svm_predictions)))
    print("MCC:", str(sklearn.metrics.matthews_corrcoef(y, svm_predictions)))
    print("Precision:", str(sklearn.metrics.precision_score(y, svm_predictions)))
    print("Recall:", str(sklearn.metrics.recall_score(y, svm_predictions)))
    print("Real positive mean:", str(y.mean()))
    print("Predicted positive mean:", str(svm_predictions.mean()))
    print("Svm score:", str(pd.DataFrame(np.asarray(svm_scores)).describe()))


def test_all_algorithms(test_name: str, data: dict, output="summary"):
    names = ["Linear SVM", "Decision Tree", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]

    classifiers = [
        SVC(kernel="linear", C=1),
        DecisionTreeClassifier(max_depth=5),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    summary = pd.DataFrame()

    X = data["features"]
    y = data["target"]
    kf = KFold(len(X), n_folds=len(X))

    for name, clf in zip(names, classifiers):
        print("Testing on ", name, "for", test_name)
        predictions = []
        scores = []
        for k, (train, test) in enumerate(kf):
            clf.fit(X[train], y[train])
            score = clf.score(X[test], y[test])
            prediction = clf.predict(X[test])
            scores.append(score)
            predictions.extend(prediction)

        predictions = np.asarray(predictions)
        scores = np.asarray(scores)

        result = {}
        result["F1"] = sklearn.metrics.f1_score(y, predictions)
        result["MCC"] = sklearn.metrics.matthews_corrcoef(y, predictions)
        result["Precision"] = sklearn.metrics.precision_score(y, predictions)
        result["Recall"] = sklearn.metrics.recall_score(y, predictions)
        result["Real positive mean"] = y.mean()
        result["Predicted positive mean"] = predictions.mean()
        result["Average score"] = scores.mean()

        summary[name] = pd.Series(result)

    summary.to_csv(output + ".csv")
    summary.to_latex(output + ".tex")
