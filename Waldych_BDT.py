import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

models_dir = '/eos/user/s/swaldych/smart_pix/labels/models'

sensor_geom = "50x12P5x150_0fb"
threshold = 0.2 #in GeV
tag = f"{sensor_geom}_0P{str(threshold - int(threshold))[2:]}thresh"

dfx = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/FullPrecisionInputTrainSet_{tag}.csv") #y-local
dfy = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetLabel_{tag}.csv") 
pt=pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetPt_{tag}.csv")

x = dfx.values
y = dfy.values
real_pt=pt.values

X_train, X_test, y_train, y_test, pt_train, pt_test = train_test_split(
    X, y, real_pt, test_size=0.2, shuffle=True, random_state=42) #we are saying split arrays the same way

#  Convert data to XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "multi:softprob",  # "softmax option produces a trained Booster object whose predict method returns a 1d array of predicted labels, whereas the softprob option produces a trained Booster object whose predict method returns a 2d array of predicted probabilities"
    "eval_metric": "auc",  # AUC for evaluation
    "max_depth": 5,  # Tree depth
    "eta": 0.05,  # Learning rate
    "seed": 42,  # Reproducibility
    "num_class":3 #three classes...0,1,2 
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=10,
    verbose_eval=10  # Print progress every 10 iterations
)
#----------------------------------------------------------
#  Predictions
y_pred_proba = model.predict(dtest)  # Probabilities
y_pred = y_pred_proba.argmax(axis=1)

accepted = (y_pred == 0)

pt_vals = []
acc_vals = []

step = 0.2   # GeV
pmin = pt_test.min()
pmax = pt_test.max()

p = pmin
while p < pmax:

    total = 0
    passed = 0

    for i in range(len(pt_test)):
        if p <= pt_test[i] < p + step:
            total += 1
            if accepted[i]:
                passed += 1

    if total > 0:
        pt_vals.append(p + step/2)
        acc_vals.append(passed / total)
        err = np.sqrt(p * (1 - p) / total)

    p += step


plt.errorbar(pt_vals, acc_vals,err,fmt='o',markersize=3)
plt.xlabel("true pt (GeV)")
plt.title("Model 2: Classifier acceptance as a function of pT")
plt.ylabel("classifier acceptance pT > |0.2| GeV")
plt.ylim(0,1)
plt.show()

#-----------------------------------------------------------

#  Evaluation
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(
    y_test,
    y_pred_proba,
    multi_class="ovr",
    average="macro"
)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

model.save_model(f"{models_dir}/xgboost_model_{tag}.json")

print("Model saved to xgboost_model.json")

#  Plot & Save ROC Curve
signal_class = 0

y_test_bin = (y_test == signal_class).astype(int)
y_score = y_pred_proba[:, signal_class]

fpr, tpr, _ = roc_curve(y_test_bin, y_score)

plt.figure(figsize=(3.5, 2.5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random chance
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: High pT v (both) low pT")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")  #  Save as PNG
plt.close()  #  Close the figure to avoid display

#------------------------------------------------

