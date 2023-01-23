#! /usr/bin/env python3
#
# From the output of the make.dataframe.py,
# Build a Pandas dataframe and classify eddies which last at least --duration
# doing a grid search for the hyperparameters of a Random Forest classifier
#
# Jan-2023, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import pandas as pd
import numpy as np
from BuildDataFrame import mkDataFrame, shuffleDataFrame, splitDataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
parser.add_argument("--fracTest", type=float, default=0.25, help="Fraction to reserve for testing")
args = parser.parse_args()

rng = np.random.default_rng(args.seed) 

df = mkDataFrame(args.input, args.duration)
df = shuffleDataFrame(rng, df) # Shuffle the rows randomly
df["qAntiCyclonic"] = np.logical_not(df.qCyclonic)
(dfTrn, dfTst) = splitDataFrame(rng, df, args.fracTest)
print("Shapes", df.shape, dfTrn.shape, dfTst.shape)

items = []
for key in sorted(df.columns):
    if key in ("qPersistent", "date", "duration", "distPerDay_slope"): continue
    # if key.endswith("_mean"): continue
    # if key.endswith("_median"): continue
    # if key.endswith("_sigma"): continue
    # if key.endswith("_slope"): continue
    items.append(key)
    # print(key)

xTrn = dfTrn[items]
xTst = dfTst[items]
yTrn = dfTrn.qPersistent
yTst = dfTst.qPersistent
print("Persistant training", yTrn.sum(), yTrn.sum()/yTrn.size, yTst.sum(), yTst.sum()/yTst.size)

key = "Random Forest"

pipeline = Pipeline(steps=[
    ("std", StandardScaler()),
    ("clf", RandomForestClassifier()),
    ])

# myScorer = "accuracy"
# myScorer = "balanced_accuracy"
# myScorer = "top_k_accuracy"
# myScorer = "average_precision"
# myScorer = "neg_brier_score"
# myScorer = "f1"
# myScorer = "neg_log_loss"
myScorer = "precision" # TP / (TP + FP)
# myScorer = "recall" # TP / (TP + FN)
# myScorer = "jaccard"
# myScorer = "roc_auc"
# myScorer = "roc_auc_ovr"
# myScorer = "roc_auc_ovo"
# myScorer = "roc_auc_ovo_weighted"

clf = GridSearchCV(estimator=pipeline,
                   scoring=myScorer,
                   param_grid={
                       "clf__max_depth": np.arange(15, 31, 5),
                       "clf__n_estimators": np.arange(15, 61, 5),
                       "clf__max_features": ["sqrt", "log2"],
                       },
                   )

clf.fit(xTrn, yTrn)
    # score = clf.score(xTst, yTst)
    # print(key, score)
clfParams = clf.best_params_;
for k in sorted(clfParams): print(k, clfParams[k])
yPred = clf.predict(xTst)
    # print(classification_report(yTst, yPrd))

cm = confusion_matrix(yTst, yPred)
print("{:4.1f}% FP {:4.1f}% TP {:4.1f}% Positive {}\nscorer {} duration {}".format(
    cm[0,1] / (cm[0,0] + cm[0,1]) * 100,
    cm[1,1] / (cm[1,0] + cm[1,1]) * 100,
    cm[1,1] / (cm[0,1] + cm[1,1]) * 100,
    clf.score(xTst, yTst), 
    key,
    myScorer, args.duration))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_,
                              )
plt.show()

# plot_confusion_matrix(clf, xTst, yTst)
# plt.show()
# for item in sorted(clf.cv_results.keys()): print(item)
