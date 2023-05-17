# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
"""Read all experiments in dir."""
# + tags=[]
import os
import pickle

import pandas as pd

results_dir = "results/simulations/ExAE-smart/"

df = pd.DataFrame()

for experiment in os.listdir(results_dir):
    if experiment[:3] != "exp":
        continue
    params_file = os.path.join(results_dir, experiment, "params.pkl")
    with open(params_file, "rb") as f:
        parameters = pickle.load(f)
    df = pd.concat(
        [df, pd.DataFrame([pd.Series(parameters)], index=[experiment])]
    )

# + tags=[]
df

# + tags=[]
pd.DataFrame([pd.Series(parameters)], index=[experiment])
# -
