from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.datasets import *
from joblib import dump, load

pkl_dir = Path.cwd() / "pkls"
data_dir = Path.cwd()/"data"

# regression
X_test = pd.read_csv(data_dir/"xvalues.csv")
y_test = pd.read_csv(data_dir/"yvalues.csv")


model = load(pkl_dir/"model_ed.joblib")

rf_explainer = RegressionExplainer(
    model, X_test, y_test, precision='float32')
db = ExplainerDashboard(
    rf_explainer, title="Bulldozer Blue Book", shap_interaction=False, whatif=False, hide_treepathgraph=True)
db.to_yaml("dashboard.yaml", explainerfile=pkl_dir/"explainer.joblib",
           dump_explainer=True)
