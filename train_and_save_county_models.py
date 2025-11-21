import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

# ----------------------------
# Load & clean data
# ----------------------------
housingdata = pd.read_csv("/Users/rox/Desktop/sta160 api housing/housingvars.csv")

counties = ["Los Angeles", "Fresno", "San Diego", "Sacramento", 
            "San Francisco", "Santa Clara", "Alameda"]

housingdata = housingdata[housingdata["County"].isin(counties)].copy()
housingdata = housingdata.drop(columns=['City', 'State'], errors='ignore')

# Split each county
for county in counties:
    df = housingdata[housingdata['County'] == county]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    globals()[f"{county.replace(' ','_')}_train"] = train_df.reset_index(drop=True)
    globals()[f"{county.replace(' ','_')}_test"]  = test_df.reset_index(drop=True)

# Clean SF
sf_train = globals()['San_Francisco_train'].drop(columns=['Population'])
sf_test = globals()['San_Francisco_test'].drop(columns=['Population'])

num_cols_train = sf_train.select_dtypes(include='number').columns
num_cols_test = sf_test.select_dtypes(include='number').columns

sf_train[num_cols_train] = sf_train[num_cols_train].fillna(0)
sf_test[num_cols_test] = sf_test[num_cols_test].fillna(0)

globals()['San_Francisco_train'] = sf_train
globals()['San_Francisco_test'] = sf_test

# Other counties dropna
other = ["Los_Angeles","Fresno","San_Diego","Sacramento","Santa_Clara","Alameda"]
for c in other:
    globals()[f"{c}_train"] = globals()[f"{c}_train"].dropna()
    globals()[f"{c}_test"] = globals()[f"{c}_test"].dropna()

# ----------------------------
# Evaluate and train
# ----------------------------
def evaluate_model(model, counties, gdict, log_target=True, winsor_q=0.99):

    models = {}
    feature_columns = None  # all counties share same numeric-only columns

    for county in counties:
        key = county.replace(" ", "_")
        train_df = gdict[f"{key}_train"]
        test_df = gdict[f"{key}_test"]

        X_train = train_df.drop(columns=['Price', 'County'], errors='ignore')\
                          .select_dtypes(include=['float64','int64'])
        X_test = test_df[X_train.columns]

        if feature_columns is None:
            feature_columns = list(X_train.columns)

        y_train = train_df['Price'].values
        y_test = test_df['Price'].values

        # Apply winsor + log
        if log_target:
            y_train = np.log1p(np.clip(y_train, None, np.quantile(y_train, winsor_q)))

        mdl = clone(model)
        mdl.fit(X_train, y_train)
        models[county] = mdl

    return models, feature_columns


# Train RF with same parameters as your code
rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=42)
county_models, feature_columns = evaluate_model(rf, counties, globals())

# Save dict
save_dict = {
    "models": county_models,
    "feature_columns": feature_columns,
    "counties": counties,
    "log_target": True,
    "winsor_q": 0.99
}

joblib.dump(save_dict, "county_models.joblib")
print("Saved county_models.joblib")
