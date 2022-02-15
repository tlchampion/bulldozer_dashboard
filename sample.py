import random

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.datasets import *
from joblib import dump, load

pkl_dir = Path.cwd() / "pkls"
data_dir = Path.cwd()/"data"

# regression
X_test = pd.read_csv(data_dir/"Xtest_model_ed.csv")
y_test = pd.read_csv(data_dir/"ytest_model_ed.csv")


X_test_small = X_test[0:50]
y_test_small = y_test[0:50]

idx = X_test['YearMade'] > 1900

X_test = X_test[idx]
y_test = y_test[idx]

choices = X_test.index.tolist()
print(y_test.shape)

idx2 = random.sample(choices, 50)
print(idx2)

X_test = X_test[X_test.index.isin(idx2)]
y_test = y_test[y_test.index.isin(idx2)]

print(X_test.shape, y_test.shape)

Xvalues = X_test.to_csv("./data/xvalues.csv", index=False)
yvalues = y_test.to_csv("./data/yvalues.csv", index=False)
