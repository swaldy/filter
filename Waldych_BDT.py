from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sensor_geom = "50x12P5x150_0fb"
threshold = 0.2 #in GeV
tag = f"{sensor_geom}_0P{str(threshold - int(threshold))[2:]}thresh"

dfx = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/FullPrecisionInputTrainSet_{tag}.csv") #y-local
dfy = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetLabel_{tag}.csv") 
pt=pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetPt_{tag}.csv")

x = dfx.values
y = dfy.values
real_pt=pt.values

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True)

bst = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.001, objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
print(type(preds))
print(preds)

pred_class = np.argmax(preds, axis=0) #returns the indices of the maximum values along the rows (axis=0 gives col)
print(pred_class)

# print("pred_class counts:", np.bincount(pred_class, minlength=3))
# print("overall acceptance (pred==0):", np.mean(pred_class == 0))

# accepted = (pred_class == 0)

# pt_vals = []
# acc_vals = []

# step = 0.2   # GeV
# pmin = pt_test.min()
# pmax = pt_test.max()

# p = pmin
# while p < pmax:

#     total = 0
#     passed = 0

#     for i in range(len(pt_test)):
#         if p <= pt_test[i] < p + step:
#             total += 1
#             if accepted[i]:
#                 passed += 1

#     if total > 0:
#         pt_vals.append(p + step/2)
#         acc_vals.append(passed / total)
#         err = np.sqrt(p * (1 - p) / total)

#     p += step


# plt.errorbar(pt_vals, acc_vals,err,fmt='o',markersize=3)
# plt.xlabel("true pt (GeV)")
# plt.title("Model 2: Classifier acceptance as a function of pT")
# plt.ylabel("classifier acceptance pT > |0.2| GeV")
# plt.ylim(0,1)
# plt.show()
