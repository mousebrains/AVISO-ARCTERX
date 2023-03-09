#! /usr/bin/env python3
#
# From the output of the make.dataframe.py,
# Build a Pandas dataframe and classify eddies which last at least --duration
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BuildDataFrame import mkDataFrame, splitDataFrame, shuffleDataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import os

parser = ArgumentParser()
parser.add_argument("input", type=str, help="Input NetCDF files with dataframe data.")
parser.add_argument("--duration", type=int, default=75,
                    help="Number of days after measurement date to call an eddy long lived")
parser.add_argument("--seed", type=int, default=123456789, help="Random number seed")
parser.add_argument("--fracTest", type=float, default=0.25, help="Fraction of dataset for testing")
args = parser.parse_args()

rng = np.random.default_rng(args.seed) 

classifiers = {
#        "Nearest Neighbors 3": KNeighborsClassifier(3),
#        "Nearest Neighbors 5": KNeighborsClassifier(5),
#        "Nearest Neighbors 10": KNeighborsClassifier(10),
#        "Nearest Neighbors 100": KNeighborsClassifier(100),
        # "Linear SVM": SVC(kernel="linear", C=0.025), # scalar divide
#        "Poly SVM": SVC(kernel="poly", C=0.25),
#        "Sigmoid SVM": SVC(kernel="sigmoid", C=0.25),
        # "RBF SVM": SVC(gamma=2, C=1), # scalar divide
        # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)), # Long time
#        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        # Scalar divide
        # "Random Forest 2,5,sqrt": \
                # RandomForestClassifier(max_depth=2, n_estimators=5, max_features="sqrt"),
#        "Random Forest 5,10,sqrt": \
#                RandomForestClassifier(max_depth=5, n_estimators=10, max_features="sqrt"),
        "Random Forest 10,20,sqrt": \
                RandomForestClassifier(max_depth=10, n_estimators=20, max_features="sqrt"),
#        "Random Forest 15,30,sqrt": \
#                RandomForestClassifier(max_depth=15, n_estimators=30, max_features="sqrt"),
#        "Random Forest 20,40,sqrt": \
#                RandomForestClassifier(max_depth=20, n_estimators=40, max_features="sqrt"),
#        "Random Forest 40,80,sqrt": \
#                RandomForestClassifier(max_depth=40, n_estimators=80, max_features="sqrt"),
        # "NeuralNet": MLPClassifier(alpha=1, max_iter=1000),
#        "AdaBoost": AdaBoostClassifier(),
#        "Naive Bayes": GaussianNB(),
        # "QDA": QuadraticDiscriminantAnalysis(), # Something is colinear
        }

df = mkDataFrame(args.input, args.duration)
lastDate = df.date.max()
print("Max Date", lastDate)
df = shuffleDataFrame(rng, df) # Shuffle the rows randomly
df["qAntiCyclonic"] = np.logical_not(df.qCyclonic)
dfLast = df[df.date == lastDate]
df = df[df.date != lastDate]
(dfTrn, dfTst) = splitDataFrame(rng, df, args.fracTest)
nYearsTrn = np.unique(dfTrn.date.astype("datetime64[Y]")).shape[0]
nYearsTst = np.unique(dfTst.date.astype("datetime64[Y]")).shape[0]
print("Shapes", df.shape, dfTrn.shape, dfTst.shape, dfLast.shape)
print("nYears", nYearsTrn, nYearsTst)


items = []
for key in sorted(df.columns):
    if key in ("qPersistent", "date", "duration", "distPerDay_slope"): continue
    # if key.endswith("_mean"): continue
    # if key.endswith("_median"): continue
    # if key.endswith("_sigma"): continue
    # if key.endswith("_slope"): continue
    items.append(key)

xTrn = dfTrn[items]
xTst = dfTst[items]
xLast = dfLast[items]
yTrn = dfTrn.qPersistent
yTst = dfTst.qPersistent
print("Persistant fraction training", yTrn.sum(), yTrn.sum()/yTrn.size,
      "testing", yTst.sum(), yTst.sum()/yTst.size)

for key in classifiers:
    clf = make_pipeline(StandardScaler(), classifiers[key])
    print(clf)
    clf.fit(xTrn, yTrn)
    # score = clf.score(xTst, yTst)
    # print(key, score)
    yPred = clf.predict(xTst)
    print(classification_report(yTst, yPred))
    cm = confusion_matrix(yTst, yPred)
    print("{:4.1f}% FP {:4.1f}% TP {:4.1f}% Positive {} nPositive {}/year".format(
        cm[0,1] / (cm[0,0] + cm[0,1]) * 100,
        cm[1,1] / (cm[1,0] + cm[1,1]) * 100,
        cm[1,1] / (cm[0,1] + cm[1,1]) * 100,
        key,
        (cm[0,1] + cm[1,1]) / nYearsTst))

    # print(key, "\n{:4d} {:4d} {:4.1f}% FP\n{:4d} {:4d} {:4.1f}% TP {:.1f}% Positive".format(
        # cm[0,0], cm[0,1],
        # cm[0,1] / (cm[0,0] + cm[0,1]) * 100,
        # cm[1,0], cm[1,1],
        # cm[1,1] / (cm[1,0] + cm[1,1]) * 100,
        # cm[1,1] / (cm[0,1] + cm[1,1]) * 100,
        # ))
    # cmDisplay = ConfusionMatrixDisplay(cm).plot()
    # plt.title(key)
    # plt.show()

    # Classify last date
    yLast = clf.predict(xLast)
    tgt = xLast[yLast != 0]
    print("# of candidates for", lastDate.date(), "found", tgt.shape[0])
    print(tgt.loc[:,("qAntiCyclonic", "latitude", "longitude", "amplitude", "distPerDay", "effective_area", "preDays", "speed_radius")])
    # for item in sorted(tgt.columns): print(item)

