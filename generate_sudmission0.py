""" Script to generate the initial Kaggle submission. """
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# to make the output stable across runs
np.random.seed(42)

# load the data
def load_housing_data(housing_path="data/"):
    """ Load the training and test housing data. """
    train_path = os.path.join(housing_path, "train.csv")
    test_path = os.path.join(housing_path, "test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path) 

def main():
    """ Main function. """
    housing, housing_test = load_housing_data()

    # separate to labels and numerical/categorical attributes
    housing_labels = housing["SalePrice"].copy()
    housing_num = housing.select_dtypes(include=[np.number]).drop("SalePrice", axis=1)
    housing_cat = housing.select_dtypes(include="object")
    attribs_num = list(housing_num)
    attribs_cat = list(housing_cat)

    # numerical data pipeline
    pipeline_num = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    # categorical data pipeline
    # TODO


    full_pipeline = ColumnTransformer([
            ("num", pipeline_num, attribs_num),
            #("cat", OneHotEncoder(), attribs_cat),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # parametric grid search
    param_grid = [
        # try 12 (3x4) combinations of hyperparameters
        {'n_estimators': [25, 30, 35], 'max_features': [4, 8, 12, 16]},
        # then try 6 (2x3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [25, 30, 35], 'max_features': [6, 8, 12]},
      ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    grid_search.best_params_

    # final model and submission
    final_model = grid_search.best_estimator_
    X_test = housing_test
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    # create submission CSV file
    submission = X_test[["Id"]].copy()
    submission["SalePrice"] = final_predictions
    submission.to_csv("data/submission0.csv", index=False)

if __name__ == "__main__":
    main()

