
# xgboost is a dependency of dtreeviz, but too large (>350M) for heroku
# so we uninstall it and mock it here:
from explainerdashboard import *
# from dash_bootstrap_components.themes import FLATLY, BOOTSTRAP  # bootstrap theme
import dash
from flask import Flask
from pathlib import Path
from unittest.mock import MagicMock
import sys
# sys.modules["xgboost"] = MagicMock()


pkl_dir = Path.cwd() / "pkls"


db = ExplainerDashboard.from_config(pkl_dir/"explainer.joblib",
                                    "dashboard.yaml", title="Bulldozer Bluebook")

app = db.flask_server()
