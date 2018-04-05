from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.externals import joblib

X, y = load_iris(return_X_y=True)
clr = SVC()
clr.fit(X, y)

joblib.dump(clr, "svc_model.pkl")
