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
import matplotlib.pyplot as plt
from BuildDataFrame import mkDataFrame, splitDataFrame, shuffleDataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
import skops.io as sio

def myScoreFunc(yTrue, yPred):
    tp = np.logical_and(yPred, yTrue).sum()
    # tn = np.logical_and(np.logical_not(yPred), np.logical_not(yTrue)).sum()
    # fn = np.logical_and(yPred, np.logical_not(yTrue)).sum()
    fp = np.logical_and(np.logical_not(yPred), yTrue).sum()
    return tp / (tp + fp)

parser = ArgumentParser()
parser.add_argument("input", type=str, help="Input NetCDF files with dataframe data.")
parser.add_argument("--duration", type=int, default=75,
                    help="Number of days after measurement date to call an eddy long lived")
parser.add_argument("--seed", type=int, default=123456789, help="Random number seed")
parser.add_argument("--fracTest", type=float, default=0.25, help="Fraction of dataset for testing")
parser.add_argument("--output", type=str, help="Model output filename")
args = parser.parse_args()

rng = np.random.default_rng(args.seed) 

df = mkDataFrame(args.input, args.duration)
lastDate = np.datetime64(df.date.max()).astype("datetime64[D]")
print("Last Date", lastDate)
df = shuffleDataFrame(rng, df) # Shuffle the rows randomly
df["qAntiCyclonic"] = np.logical_not(df.qCyclonic)
dfLast = df[df.date == lastDate]
df = df[df.date != lastDate]
(dfTrn, dfTst) = splitDataFrame(rng, df, args.fracTest)
nYearsTrn = np.unique(dfTrn.date.astype("datetime64[Y]")).shape[0]
nYearsTst = np.unique(dfTst.date.astype("datetime64[Y]")).shape[0]
print("Shapes", df.shape, dfTrn.shape, dfTst.shape, dfLast.shape)
print("nYears training", nYearsTrn, "testing", nYearsTst)

items = []

# Original
# "qPersistent", "date", "duration", "distPerDay_slope"
toDrop = (
        "qPersistent",
        "date",
        "duration",
        "distPerDay_slope",
        "oni",
        # "KE",
        # "PE",
        # "amplitude",
        # "antiDistance",
        # "cyclDistance",
        # "distPerDay",
        # "effective_area",
        # "effective_contour_height",
        # "effective_contour_shape_error",
        # "effective_radius",
        # "inner_contour_height",
        # "latitude",
        # "latitude_max",
        # "longitude",
        # "longitude_max",
        # "preDays",
        # "qAntiCyclonic",
        # "qCyclonic",
        # "speed_area",
        # "speed_average",
        # "speed_contour_height",
        # "speed_contour_shape_error",
        # "speed_radius",
        )

for key in sorted(df.columns):
    if key in toDrop: continue
    # if key.endswith("_mean"): continue
    # if key.endswith("_median"): continue
    # if key.endswith("_sigma"): continue
    # if key.endswith("_slope"): continue
    items.append(key)
    # print(key)

xTrn = dfTrn[items]
xTst = dfTst[items]
xLast = dfLast[items]
yTrn = dfTrn.qPersistent
yTst = dfTst.qPersistent
print("Persistant fraction training", 
      yTrn.sum(), "of", yTrn.size, "{:.2f}%".format(yTrn.sum()/yTrn.size * 100),
      "testing", 
      yTst.sum(), "of", yTst.size, "{:.2f}%".format(yTst.sum()/yTst.size * 100))

key = "Random Forest"

pipeline = Pipeline(steps=[
    ("std", StandardScaler()),
    ("clf", RandomForestClassifier()),
    ])

# For description see:
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#
# myScorer = "accuracy" # fraction correctly classified
# myScorer = "balanced_accuracy" # unbalanced partition accuracy
# myScorer = "top_k_accuracy" # number of top k labels predicted ?
# myScorer = "average_precision" # sum_n (R_n - R_(n-1)) P_n (R -> recall, P -> precision)
# myScorer = "neg_brier_score" # Mean square difference pred and actual
# myScorer = "f1" # 2 * precision * recall / (precision + recall)
# myScorer = "neg_log_loss"
# myScorer = "precision" # TP / (TP + FP)
myScorer = "recall" # TP / (TP + FN)
# myScorer = "jaccard" # Jaccard similarity coefficient score
# myScorer = "roc_auc" # Receiver Operating Characteristic curve area from pred scores
# myScorer = "roc_auc_ovr"
# myScorer = "roc_auc_ovo"
# myScorer = "roc_auc_ovo_weighted"
# myScorer = make_scorer(myScoreFunc, greater_is_better=True)

clf = GridSearchCV(estimator=pipeline,
                   scoring=myScorer,
                   param_grid={
                       "clf__max_depth": np.arange(15, 36, 5),
                       "clf__n_estimators": np.arange(10, 61, 5),
                       "clf__max_features": ["sqrt", "log2"],
                       },
                   )

clf.fit(xTrn, yTrn)
    # score = clf.score(xTst, yTst)
    # print(key, score)
if args.output: # Save the model
    obj = sio.dump(clf, args.output)


clfParams = clf.best_params_;
for k in sorted(clfParams): print(k, clfParams[k])
yPred = clf.predict(xTst)
    # print(classification_report(yTst, yPrd))
print(classification_report(yTst, yPred))
cm = confusion_matrix(yTst, yPred)
print("{:4.1f}% FP {:4.1f}% TP {:4.1f}% Positive {} nPositive {}/year".format(
    cm[0,1] / (cm[0,0] + cm[0,1]) * 100,
    cm[1,1] / (cm[1,0] + cm[1,1]) * 100,
    cm[1,1] / (cm[0,1] + cm[1,1]) * 100,
    key,
    (cm[0,1] + cm[1,1]) / nYearsTst))

print("scorer", myScorer, "duration", args.duration, "days")

# Classify last date
yLast = clf.predict(xLast)
tgt = xLast[yLast != 0]
print("# of candidates for", lastDate, "found", tgt.shape[0])
print(tgt.loc[:,("qAntiCyclonic", "latitude", "longitude", "amplitude", "preDays", "distPerDay", "effective_radius", "speed_radius")])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_,
                              )
plt.show()

# plot_confusion_matrix(clf, xTst, yTst)
# plt.show()
# for item in sorted(clf.cv_results.keys()): print(item)
