from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
