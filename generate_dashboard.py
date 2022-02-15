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

# X_test = pd.read_csv(data_dir/"Xtest_model2.csv")
# y_test = pd.read_csv(data_dir/"ytest_model2.csv")

X_test_small = X_test[0:50]
y_test_small = y_test[0:50]

model = load(pkl_dir/"model_ed.joblib")
# model = load(pkl_dir/"model2_best.joblib")
rf_explainer = RegressionExplainer(
    model, X_test_small, y_test_small, precision='float32')
db = ExplainerDashboard(
    rf_explainer, title="Bulldozer Blue Book", shap_interaction=False, whatif=False, hide_treepathgraph=True)
db.to_yaml("dashboard.yaml", explainerfile=pkl_dir/"explainer.joblib",
           dump_explainer=True)
