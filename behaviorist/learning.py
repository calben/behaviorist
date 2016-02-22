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
    print(df.describe())
    data = {}
    data["target"] = df["label"].values
    data["features"] = df.drop("label", axis=1).values
    return data


def filter_feature_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    mat = mat.dropna(axis=0)
    mat = mat[mat["distance-mean"] != 0]
    mat = mat.reset_index(drop=True)
    print(mat)
    mat = mat[["label", "distance-std", "distance-75%"]]
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


def test_all_algorithms(data: dict):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    summary = pd.DataFrame()

    X = data["features"]
    y = data["target"]
    kf = KFold(len(X), n_folds=len(X))

    for name, clf in zip(names, classifiers):
        predictions = []
        scores = []
        for k, (train, test) in enumerate(kf):
            clf.fit(X[train], y[train])
            score = clf.score(X[test], y[test])
            prediction = clf.predict(X[test])
            scores.append(score)
            predictions.append(prediction)

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

    summary.to_csv("summary.csv")
